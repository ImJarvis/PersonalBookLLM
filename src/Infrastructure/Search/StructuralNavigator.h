#pragma once
#include "Core/Interfaces/IStructuralIndexer.h"
#include "Core/Models/SearchResult.h"
#include <memory>
#include <vector>
#include <string>

namespace LocalNotebookLLM::Infrastructure {

    class DatabaseManager;

    /// @brief Navigates the structural document tree to find contextually relevant sections.
    /// Used as Tier 2 in the hybrid search pipeline when BM25 returns low-confidence results.
    ///
    /// Strategy:
    /// 1. Extract keywords from query
    /// 2. Walk heading nodes, scoring by keyword overlap in heading text
    /// 3. For matched headings, collect child content as context
    /// 4. Score by structural relevance (heading-boost heuristic)
    class StructuralNavigator {
    public:
        explicit StructuralNavigator(std::shared_ptr<DatabaseManager> db);

        /// Navigate the structural tree to find sections matching the query.
        [[nodiscard]]
        std::vector<Core::SearchResult>
        Navigate(const std::string& query, int topK = 10) const;

    private:
        std::shared_ptr<DatabaseManager> m_db;

        /// Score a heading's relevance to the query keywords.
        [[nodiscard]]
        static double ScoreHeadingRelevance(const std::string& headingText,
                                            const std::vector<std::string>& keywords);
    };

} // namespace LocalNotebookLLM::Infrastructure
