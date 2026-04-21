#include "Infrastructure/Storage/SqliteDocumentRepository.h"
#include "Core/Enums/DocumentFormat.h"

namespace LocalNotebookLLM::Infrastructure {

    SqliteDocumentRepository::SqliteDocumentRepository(std::shared_ptr<DatabaseManager> db)
        : m_db(std::move(db)) {}

    std::expected<int64_t, std::string>
    SqliteDocumentRepository::Add(const Core::DocumentMetadata& doc) {
        try {
            auto stmt = m_db->Prepare(R"(
                INSERT INTO documents (filename, filepath, format, page_count, node_count, file_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            )");
            stmt.Bind(1, doc.filename);
            stmt.Bind(2, doc.filepath);
            stmt.Bind(3, Core::DocumentFormatToString(doc.format));
            stmt.Bind(4, doc.pageCount);
            stmt.Bind(5, static_cast<int>(doc.nodeCount));
            stmt.Bind(6, doc.fileHash);
            stmt.Execute();
            return m_db->LastInsertRowId();
        } catch (const std::exception& e) {
            return std::unexpected(std::string("Failed to add document: ") + e.what());
        }
    }

    void SqliteDocumentRepository::Remove(int64_t documentId) {
        auto stmt = m_db->Prepare("DELETE FROM documents WHERE id = ?");
        stmt.Bind(1, documentId);
        stmt.Execute();
    }

    std::vector<Core::DocumentMetadata> SqliteDocumentRepository::ListAll() const {
        auto stmt = m_db->Prepare(
            "SELECT id, filename, filepath, format, page_count, node_count, ingested_at, file_hash "
            "FROM documents ORDER BY ingested_at DESC");

        std::vector<Core::DocumentMetadata> results;
        while (stmt.Step()) {
            results.push_back(RowToMetadata(stmt));
        }
        return results;
    }

    std::expected<Core::DocumentMetadata, std::string>
    SqliteDocumentRepository::GetById(int64_t documentId) const {
        auto stmt = m_db->Prepare(
            "SELECT id, filename, filepath, format, page_count, node_count, ingested_at, file_hash "
            "FROM documents WHERE id = ?");
        stmt.Bind(1, documentId);

        if (!stmt.Step()) {
            return std::unexpected("Document not found: id=" + std::to_string(documentId));
        }
        return RowToMetadata(stmt);
    }

    bool SqliteDocumentRepository::ExistsByHash(const std::string& fileHash) const {
        auto stmt = m_db->Prepare("SELECT COUNT(*) FROM documents WHERE file_hash = ?");
        stmt.Bind(1, fileHash);
        stmt.Step();
        return stmt.GetInt(0) > 0;
    }

    Core::DocumentMetadata SqliteDocumentRepository::RowToMetadata(SqliteStatement& stmt) {
        Core::DocumentMetadata meta;
        meta.id         = stmt.GetInt64(0);
        meta.filename   = stmt.GetString(1);
        meta.filepath   = stmt.GetString(2);
        meta.format     = Core::ExtensionToFormat("." + stmt.GetString(3)); // stored as "PDF"/"DOCX"
        meta.pageCount  = stmt.GetInt(4);
        meta.nodeCount  = static_cast<size_t>(stmt.GetInt(5));
        meta.ingestedAt = stmt.GetString(6);
        meta.fileHash   = stmt.GetString(7);

        // Fix format mapping: stored as "PDF" not ".pdf"
        std::string fmtStr = stmt.GetString(3);
        if (fmtStr == "PDF")  meta.format = Core::DocumentFormat::PDF;
        else if (fmtStr == "DOCX") meta.format = Core::DocumentFormat::DOCX;
        else if (fmtStr == "TXT")  meta.format = Core::DocumentFormat::TXT;
        else if (fmtStr == "EPUB") meta.format = Core::DocumentFormat::EPUB;

        return meta;
    }

} // namespace LocalNotebookLLM::Infrastructure
