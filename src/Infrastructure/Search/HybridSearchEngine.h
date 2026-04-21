#pragma once
#include "Core/Interfaces/IHybridSearch.h"
#include "Core/Interfaces/IStructuralIndexer.h"
#include "Core/Interfaces/IEmbeddingProvider.h"
#include "Infrastructure/Search/StructuralNavigator.h"
#include <memory>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief 3-tier search engine: BM25 → Structural Navigation → Embedding fallback.
    /// Implements IHybridSearch, cascading through tiers based on confidence thresholds.
    class HybridSearchEngine : public Core::IHybridSearch {
    public:
        HybridSearchEngine(
            std::shared_ptr<Core::IStructuralIndexer> fts5Indexer,
            std::shared_ptr<StructuralNavigator> navigator,
            std::shared_ptr<Core::IEmbeddingProvider> embeddingProvider = nullptr,
            std::shared_ptr<DatabaseManager> db = nullptr
        );

        [[nodiscard]]
        std::vector<Core::HybridSearchResult>
        Search(const std::string& query, const Core::HybridSearchOptions& opts = {}) override;

    private:
        std::shared_ptr<Core::IStructuralIndexer>  m_fts5Indexer;
        std::shared_ptr<StructuralNavigator>       m_navigator;
        std::shared_ptr<Core::IEmbeddingProvider>  m_embeddingProvider;
        std::shared_ptr<DatabaseManager>           m_db;

        /// Convert base SearchResults to HybridSearchResults with tier annotation.
        [[nodiscard]]
        static std::vector<Core::HybridSearchResult>
        ConvertToHybrid(const std::vector<Core::SearchResult>& results,
                        Core::SearchTier tier, int limit);

        /// Merge two result sets, deduplicating by nodeId (higher score wins).
        [[nodiscard]]
        static std::vector<Core::HybridSearchResult>
        MergeAndDeduplicate(const std::vector<Core::HybridSearchResult>& a,
                            const std::vector<Core::HybridSearchResult>& b);

        /// Take the top-N results from a sorted list.
        [[nodiscard]]
        static std::vector<Core::HybridSearchResult>
        TakeTop(std::vector<Core::HybridSearchResult>& results, int n);
    };

} // namespace LocalNotebookLLM::Infrastructure
