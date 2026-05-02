#pragma once
#include "Core/Interfaces/IHybridSearch.h"
#include "Core/Interfaces/IStructuralIndexer.h"
#include "Core/Interfaces/IEmbeddingProvider.h"
#include "Infrastructure/Search/StructuralNavigator.h"
#include <memory>
#include <mutex>

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

        /// Invalidate the in-memory embedding cache.
        /// Call after a document is ingested or removed so the next search reloads
        /// embeddings from SQLite. Thread-safe.
        void InvalidateCache();

    private:
        std::shared_ptr<Core::IStructuralIndexer>  m_fts5Indexer;
        std::shared_ptr<StructuralNavigator>       m_navigator;
        std::shared_ptr<Core::IEmbeddingProvider>  m_embeddingProvider;
        std::shared_ptr<DatabaseManager>           m_db;

        // ── In-memory embedding cache (M1 fix) ──
        // Loaded once from SQLite on first semantic search; reused every query.
        // Reduces Tier-3 from O(N·SQLite) to O(N) over L2/L3-cached memory.
        struct EmbeddedNode {
            int64_t             nodeId;
            int64_t             documentId;
            Core::NodeType      nodeType;
            int                 pageNumber;
            std::string         sectionPath;
            std::string         content;
            std::vector<float>  embedding;
        };
        mutable std::vector<EmbeddedNode> m_embCache;
        mutable std::mutex               m_cacheMutex;
        mutable bool                     m_cacheValid = false;

        /// (Re)load all embedded nodes from SQLite into m_embCache.
        /// Lazy — called automatically before the first semantic search.
        void LoadEmbeddingCache() const;

        /// Fetch the document's structural skeleton for summation queries:
        /// title + all headings + first paragraph under each heading.
        [[nodiscard]]
        std::vector<Core::HybridSearchResult>
        GetDocumentOverview(int64_t documentIdFilter, int topK) const;

        /// Fetch all content nodes under the heading that matches a chapter/section reference.
        [[nodiscard]]
        std::vector<Core::HybridSearchResult>
        GetChapterContent(const std::string& query, int64_t documentIdFilter, int topK) const;

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

        /// Reciprocal Rank Fusion: combines results from all 3 tiers into a single
        /// ranked list using RRF_score(doc) = Σ 1/(k + rank_in_tier_i).
        [[nodiscard]]
        static std::vector<Core::HybridSearchResult>
        FuseWithRRF(const std::vector<Core::HybridSearchResult>& bm25,
                    const std::vector<Core::HybridSearchResult>& structural,
                    const std::vector<Core::HybridSearchResult>& semantic,
                    int k = 60);
    };

} // namespace LocalNotebookLLM::Infrastructure

