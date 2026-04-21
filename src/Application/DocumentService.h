#pragma once
#include "Core/Interfaces/IDocumentParser.h"
#include "Core/Interfaces/IStructuralIndexer.h"
#include "Core/Interfaces/IDocumentRepository.h"
#include "Core/Interfaces/IMemoryOrchestrator.h"
#include "Core/Models/DocumentMetadata.h"
#include <memory>
#include <filesystem>
#include <vector>
#include <expected>
#include <functional>

namespace LocalNotebookLLM::Application {

    struct IngestionProgress {
        enum class Stage { Parsing, Validating, Indexing, Summarizing, Complete, Failed };
        Stage       stage;
        double      progressPct = 0.0;   // 0.0 - 1.0
        std::string message;
    };

    using ProgressCallback = std::function<void(const IngestionProgress&)>;

    /// @brief Orchestrates document lifecycle: ingest, remove, query metadata.
    /// Depends ONLY on interfaces — never on concrete implementations.
    class DocumentService {
    public:
        DocumentService(
            std::unique_ptr<Core::IDocumentParser> parser,
            std::shared_ptr<Core::IStructuralIndexer> indexer,
            std::shared_ptr<Core::IDocumentRepository> repository,
            std::shared_ptr<Core::IMemoryOrchestrator> memoryOrch = nullptr
        );
        ~DocumentService();

        DocumentService(const DocumentService&) = delete;
        DocumentService& operator=(const DocumentService&) = delete;
        DocumentService(DocumentService&&) noexcept;
        DocumentService& operator=(DocumentService&&) noexcept;

        [[nodiscard]]
        std::expected<Core::DocumentMetadata, std::string>
        IngestDocument(const std::filesystem::path& filePath, ProgressCallback progress = nullptr);

        void RemoveDocument(int64_t documentId);

        [[nodiscard]]
        std::vector<Core::DocumentMetadata> ListDocuments() const;

        [[nodiscard]]
        bool IsAlreadyIngested(const std::filesystem::path& filePath) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;

        static std::string ComputeFileHash(const std::filesystem::path& filePath);
    };

} // namespace LocalNotebookLLM::Application
