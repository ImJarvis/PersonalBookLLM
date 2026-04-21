#pragma once
#include "Core/Models/SearchResult.h"
#include <vector>
#include <string>

namespace LocalNotebookLLM::Core {

    struct HybridSearchOptions {
        int    topK                          = 10;
        int    documentIdFilter              = -1;
        double bm25ConfidenceThreshold       = 0.3;
        double structuralConfidenceThreshold  = 0.2;
        bool   enableEmbeddingFallback       = true;
    };

    /// @brief 3-tier search: BM25 → Structural Navigation → Embedding fallback.
    class IHybridSearch {
    public:
        virtual ~IHybridSearch() = default;

        [[nodiscard]]
        virtual std::vector<HybridSearchResult>
        Search(const std::string& query, const HybridSearchOptions& opts = {}) = 0;
    };

} // namespace LocalNotebookLLM::Core
