#pragma once
#include "Core/Interfaces/IStructuralIndexer.h"
#include "Core/Interfaces/IEmbeddingProvider.h"
#include "Infrastructure/Indexing/DatabaseManager.h"
#include <memory>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief FTS5-backed structural indexer.
    /// Walks DocumentTree, flattens into `document_nodes` table,
    /// and FTS5 triggers auto-populate the search index.
    class SqliteFts5Indexer : public Core::IStructuralIndexer {
    public:
        explicit SqliteFts5Indexer(std::shared_ptr<DatabaseManager> db, std::shared_ptr<Core::IEmbeddingProvider> embedder = nullptr);
        ~SqliteFts5Indexer() override = default;

        [[nodiscard]]
        std::expected<void, Core::IndexError>
        IndexDocument(int64_t documentId, const Core::DocumentTree& tree) override;

        void GenerateEmbeddingsBackground(int64_t documentId, std::function<void(int, int)> progressCb) override;

        void RemoveDocument(int64_t documentId) override;

        [[nodiscard]]
        std::vector<Core::SearchResult>
        Search(const std::string& query, const Core::SearchOptions& options = {}) const override;

        [[nodiscard]]
        Core::SectionContext GetSectionContext(int64_t nodeId) const override;

        [[nodiscard]]
        size_t GetIndexedNodeCount() const override;

    private:
        std::shared_ptr<DatabaseManager> m_db;
        std::shared_ptr<Core::IEmbeddingProvider> m_embedder;

        /// Recursively walk the document tree, inserting nodes into the DB.
        void WalkAndInsert(int64_t docId, const Core::DocumentNode& node,
                           std::optional<int64_t> parentDbId,
                           std::string& sectionPath, int& sortOrder);

        /// Escape user query for safe FTS5 MATCH usage.
        static std::string EscapeFts5Query(const std::string& query);

        /// Build a SearchResult from a statement row.
        static Core::SearchResult RowToSearchResult(SqliteStatement& stmt);
    };

} // namespace LocalNotebookLLM::Infrastructure
