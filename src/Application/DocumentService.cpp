#include "Application/DocumentService.h"
#include "Core/Log.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <array>

// SHA-256 — minimal portable implementation header
// In production, swap with OpenSSL or Windows CNG
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

namespace LocalNotebookLLM::Application {

    struct DocumentService::Impl {
        std::unique_ptr<Core::IDocumentParser>       parser;
        std::shared_ptr<Core::IStructuralIndexer>    indexer;
        std::shared_ptr<Core::IDocumentRepository>   repository;
        std::shared_ptr<Core::IMemoryOrchestrator>   memoryOrch;
    };

    DocumentService::DocumentService(
        std::unique_ptr<Core::IDocumentParser> parser,
        std::shared_ptr<Core::IStructuralIndexer> indexer,
        std::shared_ptr<Core::IDocumentRepository> repository,
        std::shared_ptr<Core::IMemoryOrchestrator> memoryOrch)
        : m_impl(std::make_unique<Impl>())
    {
        m_impl->parser     = std::move(parser);
        m_impl->indexer    = std::move(indexer);
        m_impl->repository = std::move(repository);
        m_impl->memoryOrch = std::move(memoryOrch);
    }

    DocumentService::~DocumentService() = default;
    DocumentService::DocumentService(DocumentService&&) noexcept = default;
    DocumentService& DocumentService::operator=(DocumentService&&) noexcept = default;

    std::expected<Core::DocumentMetadata, std::string>
    DocumentService::IngestDocument(const std::filesystem::path& filePath, ProgressCallback progress) {
        LOG_INFO("DocService", "IngestDocument() path: " + filePath.string());

        // ─── Step 1: Validate file exists ───
        if (!std::filesystem::exists(filePath)) {
            LOG_ERROR("DocService", "File not found: " + filePath.string());
            return std::unexpected("File not found: " + filePath.string());
        }
        LOG_DEBUG("DocService", "Computing file hash...");
        std::string hash = ComputeFileHash(filePath);
        LOG_DEBUG("DocService", "Hash: " + hash);
        if (m_impl->repository->ExistsByHash(hash)) {
            LOG_WARN("DocService", "Duplicate document (hash match) — skipping");
            return std::unexpected("Document already ingested (duplicate hash)");
        }

        // ─── Step 3: Parse document ───
        if (progress) progress({IngestionProgress::Stage::Parsing, 0.1, "Parsing document..."});

        auto parseResult = m_impl->parser->ParseFile(filePath);
        if (!parseResult) {
            LOG_ERROR("DocService", "Parse failed: " + parseResult.error().message);
            return std::unexpected("Parse failed: " + parseResult.error().message);
        }

        auto& tree = parseResult.value();
        if (tree.IsEmpty()) {
            LOG_ERROR("DocService", "Document produced empty tree");
            return std::unexpected("Document produced empty tree — no text extracted");
        }
        LOG_INFO("DocService", "Parsed: " + std::to_string(tree.PageCount()) + " pages, " +
            std::to_string(tree.NodeCount()) + " nodes");

        if (progress) progress({IngestionProgress::Stage::Validating, 0.4, "Validating structure..."});

        // ─── Step 4: Store metadata ───
        Core::DocumentMetadata meta;
        meta.filename  = filePath.filename().string();
        meta.filepath  = filePath.string();
        meta.format    = Core::ExtensionToFormat(filePath.extension().string());
        meta.pageCount = tree.PageCount();
        meta.fileHash  = hash;
        meta.nodeCount = tree.NodeCount();

        auto idResult = m_impl->repository->Add(meta);
        if (!idResult) {
            return std::unexpected("Failed to store metadata: " + idResult.error());
        }
        meta.id = idResult.value();

        // ─── Step 5: Index the structural tree ───
        if (progress) progress({IngestionProgress::Stage::Indexing, 0.6, "Indexing content..."});

        auto indexResult = m_impl->indexer->IndexDocument(meta.id, tree);
        if (!indexResult) {
            LOG_ERROR("DocService", "Indexing failed: " + indexResult.error().message);
            m_impl->repository->Remove(meta.id);
            return std::unexpected("Indexing failed: " + indexResult.error().message);
        }

        LOG_INFO("DocService", "Ingestion COMPLETE: " + meta.filename + 
            " — " + std::to_string(meta.nodeCount) + " sections, " +
            std::to_string(meta.pageCount) + " pages");

        if (progress) progress({IngestionProgress::Stage::Complete, 1.0,
            "Indexed " + std::to_string(meta.nodeCount) + " sections across " +
            std::to_string(meta.pageCount) + " pages"});

        return meta;
    }

    void DocumentService::RemoveDocument(int64_t documentId) {
        m_impl->indexer->RemoveDocument(documentId);
        m_impl->repository->Remove(documentId);
    }

    std::vector<Core::DocumentMetadata> DocumentService::ListDocuments() const {
        return m_impl->repository->ListAll();
    }

    bool DocumentService::IsAlreadyIngested(const std::filesystem::path& filePath) const {
        if (!std::filesystem::exists(filePath)) return false;
        std::string hash = ComputeFileHash(filePath);
        return m_impl->repository->ExistsByHash(hash);
    }

    std::string DocumentService::ComputeFileHash(const std::filesystem::path& filePath) {
#ifdef _WIN32
        // Use Windows CNG (BCrypt) for SHA-256
        std::ifstream file(filePath, std::ios::binary);
        if (!file) return "";

        BCRYPT_ALG_HANDLE hAlg = nullptr;
        BCRYPT_HASH_HANDLE hHash = nullptr;
        BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
        BCryptCreateHash(hAlg, &hHash, nullptr, 0, nullptr, 0, 0);

        std::array<char, 8192> buffer;
        while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
            BCryptHashData(hHash, reinterpret_cast<PUCHAR>(buffer.data()),
                           static_cast<ULONG>(file.gcount()), 0);
        }

        std::array<UCHAR, 32> hashBytes;
        BCryptFinishHash(hHash, hashBytes.data(), static_cast<ULONG>(hashBytes.size()), 0);

        BCryptDestroyHash(hHash);
        BCryptCloseAlgorithmProvider(hAlg, 0);

        std::ostringstream hex;
        hex << std::hex << std::setfill('0');
        for (auto b : hashBytes) hex << std::setw(2) << static_cast<int>(b);
        return hex.str();
#else
        // Fallback: use file size + name as pseudo-hash (replace with real SHA-256)
        auto sz = std::filesystem::file_size(filePath);
        return filePath.filename().string() + "_" + std::to_string(sz);
#endif
    }

} // namespace LocalNotebookLLM::Application
