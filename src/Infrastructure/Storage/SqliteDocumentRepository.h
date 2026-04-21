#pragma once
#include "Core/Interfaces/IDocumentRepository.h"
#include "Infrastructure/Indexing/DatabaseManager.h"
#include <memory>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief SQLite-backed document metadata repository.
    /// Provides CRUD operations for the `documents` table.
    class SqliteDocumentRepository : public Core::IDocumentRepository {
    public:
        explicit SqliteDocumentRepository(std::shared_ptr<DatabaseManager> db);
        ~SqliteDocumentRepository() override = default;

        [[nodiscard]]
        std::expected<int64_t, std::string>
        Add(const Core::DocumentMetadata& doc) override;

        void Remove(int64_t documentId) override;

        [[nodiscard]]
        std::vector<Core::DocumentMetadata> ListAll() const override;

        [[nodiscard]]
        std::expected<Core::DocumentMetadata, std::string>
        GetById(int64_t documentId) const override;

        [[nodiscard]]
        bool ExistsByHash(const std::string& fileHash) const override;

    private:
        std::shared_ptr<DatabaseManager> m_db;

        /// Map a SQLite row to DocumentMetadata.
        static Core::DocumentMetadata RowToMetadata(SqliteStatement& stmt);
    };

} // namespace LocalNotebookLLM::Infrastructure
