#pragma once
#include "Core/Models/SearchResult.h"
#include <string>
#include <vector>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief Lightweight token-overlap re-ranker for hybrid search results.
    /// Zero additional memory cost — uses string tokenization and overlap counting
    /// to verify that retrieved passages actually answer the query.
    class ReRanker {
    public:
        /// Re-rank results by computing token overlap between query and each passage.
        /// Returns results sorted by combined (original rank + overlap) score.
        /// @param query The user's original query string.
        /// @param results The candidate results from hybrid search.
        /// @param originalWeight Weight for the original RRF score [0.0, 1.0]. Default 0.7.
        /// @param overlapWeight Weight for the token overlap score [0.0, 1.0]. Default 0.3.
        [[nodiscard]]
        static std::vector<Core::HybridSearchResult> ReRank(
            const std::string& query,
            std::vector<Core::HybridSearchResult>& results,
            float originalWeight = 0.7f,
            float overlapWeight = 0.3f);

    private:
        /// Compute the fraction of query terms found in a passage [0.0, 1.0].
        [[nodiscard]]
        static float ComputeOverlap(const std::string& query, const std::string& passage);

        /// Tokenize a string into lowercase words.
        [[nodiscard]]
        static std::vector<std::string> Tokenize(const std::string& text);
    };

} // namespace LocalNotebookLLM::Infrastructure
