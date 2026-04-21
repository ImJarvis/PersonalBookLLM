#pragma once
#include <string>
#include <vector>

namespace LocalNotebookLLM::Application {

    /// @brief Extracts keywords and search terms from a user's natural language query.
    /// Used to construct FTS5 MATCH queries for BM25 search.
    class QueryAnalyzer {
    public:
        /// Extract meaningful keywords from a query, removing stopwords.
        [[nodiscard]]
        static std::vector<std::string> ExtractKeywords(const std::string& query);

        /// Build an FTS5-compatible MATCH expression from keywords.
        /// Example: "risk factors Asia" → "risk AND factors AND Asia"
        [[nodiscard]]
        static std::string BuildFts5Query(const std::vector<std::string>& keywords);

        /// Convenience: query → FTS5 match expression in one call.
        [[nodiscard]]
        static std::string Analyze(const std::string& query);

    private:
        static const std::vector<std::string>& GetStopwords();
    };

} // namespace LocalNotebookLLM::Application
