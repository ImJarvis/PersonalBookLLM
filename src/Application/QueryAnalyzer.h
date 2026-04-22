#pragma once
#include <string>
#include <vector>

namespace LocalNotebookLLM::Application {

    /// @brief Classifies the high-level intent of a user query.
    enum class QueryIntent {
        Keyword,        ///< Specific fact/keyword lookup (default).
        Summation,      ///< Broad overview: "what is this about", "summarize", "overview".
        ChapterSummary  ///< Specific chapter/section summary: "summarize chapter 1", "what is in chapter 3".
    };

    /// @brief Extracts keywords and search terms from a user's natural language query.
    /// Used to construct FTS5 MATCH queries for BM25 search.
    class QueryAnalyzer {
    public:
        /// Detect the high-level intent of the query.
        [[nodiscard]]
        static QueryIntent DetectIntent(const std::string& query);

        /// Extract meaningful keywords from a query, removing stopwords.
        [[nodiscard]]
        static std::vector<std::string> ExtractKeywords(const std::string& query);

        /// Build an FTS5-compatible MATCH expression from keywords.
        [[nodiscard]]
        static std::string BuildFts5Query(const std::vector<std::string>& keywords);

        /// Convenience: query → FTS5 match expression in one call.
        [[nodiscard]]
        static std::string Analyze(const std::string& query);

    private:
        static const std::vector<std::string>& GetStopwords();
    };

} // namespace LocalNotebookLLM::Application
