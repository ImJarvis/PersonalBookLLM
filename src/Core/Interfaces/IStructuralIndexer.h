#pragma once
#include "Core/Models/DocumentTree.h"
#include "Core/Models/SearchResult.h"
#include <cstdint>
#include <vector>
#include <string>
#include <expected>

namespace LocalNotebookLLM::Core {

    struct IndexError {
        enum class Code { DatabaseError, DocumentNotFound, DuplicateDocument };
        Code code;
        std::string message;
    };

    struct SearchOptions {
        int    topK             = 10;
        int    documentIdFilter = -1;   // -1 = search all documents
        double minScore         = 0.0;
    };

    /// @brief Indexes and searches structural document content.
    /// Backed by SQLite FTS5 for keyword-grounded retrieval.
    class IStructuralIndexer {
    public:
        virtual ~IStructuralIndexer() = default;

        [[nodiscard]]
        virtual std::expected<void, IndexError>
        IndexDocument(int64_t documentId, const DocumentTree& tree) = 0;

        virtual void GenerateEmbeddingsBackground(int64_t documentId, std::function<void(int, int)> progressCb) = 0;

        virtual void RemoveDocument(int64_t documentId) = 0;

        [[nodiscard]]
        virtual std::vector<SearchResult>
        Search(const std::string& query, const SearchOptions& options = {}) const = 0;

        [[nodiscard]]
        virtual SectionContext GetSectionContext(int64_t nodeId) const = 0;

        [[nodiscard]]
        virtual size_t GetIndexedNodeCount() const = 0;
    };

} // namespace LocalNotebookLLM::Core
