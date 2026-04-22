#include "Application/QueryAnalyzer.h"
#include <algorithm>
#include <sstream>
#include <cctype>
#include <regex>

namespace LocalNotebookLLM::Application {

    const std::vector<std::string>& QueryAnalyzer::GetStopwords() {
        // ⚠️  IMPORTANT: Only strip *grammatical* function words here.
        // DO NOT include content-bearing words like "summary", "chapter", "about",
        // "explain", "overview" — those words carry the user's intent and must
        // survive into the search query.
        static const std::vector<std::string> stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "every", "both", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "because", "but", "and", "or", "if", "while",
            "these", "those", "am", "it", "its", "i", "me", "my", "we", "our",
            "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
            // Truly empty referential words
            "this", "that", "get", "give", "tell", "show",
        };
        return stopwords;
    }

    QueryIntent QueryAnalyzer::DetectIntent(const std::string& query) {
        // Lowercase the query for pattern matching
        std::string lower = query;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        // ─── Chapter/Section summary patterns ───
        // e.g. "summary of chapter 1", "summarize chapter 3", "what is in chapter 2",
        //      "explain section 4", "overview of part 2"
        static const std::regex chapterPattern(
            R"((chapter|section|part|ch\.?)\s*(\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten))",
            std::regex::icase);
        if (std::regex_search(lower, chapterPattern)) {
            return QueryIntent::ChapterSummary;
        }

        // ─── Full-document summation patterns ───
        // e.g. "what is this about", "summarize this document", "give me an overview",
        //      "what does this pdf contain", "explain this document"
        static const std::vector<std::string> summationPhrases = {
            "what is this",
            "what is the document",
            "what is this pdf",
            "what is this book",
            "what is this about",
            "what does this",
            "what does the document",
            "summarize",
            "summarise",
            "summary",
            "overview",
            "main topic",
            "main topics",
            "about this",
            "about the document",
            "explain this",
            "key points",
            "key ideas",
            "main points",
            "what are the topics",
            "what topics",
            "table of contents",
            "contents of",
        };
        for (const auto& phrase : summationPhrases) {
            if (lower.find(phrase) != std::string::npos) {
                return QueryIntent::Summation;
            }
        }

        return QueryIntent::Keyword;
    }

    std::vector<std::string> QueryAnalyzer::ExtractKeywords(const std::string& query) {
        std::vector<std::string> keywords;
        const auto& stopwords = GetStopwords();

        std::istringstream stream(query);
        std::string word;

        while (stream >> word) {
            // Lowercase
            std::transform(word.begin(), word.end(), word.begin(),
                           [](unsigned char c) { return std::tolower(c); });

            // Strip punctuation from edges
            while (!word.empty() && std::ispunct(static_cast<unsigned char>(word.front())))
                word.erase(word.begin());
            while (!word.empty() && std::ispunct(static_cast<unsigned char>(word.back())))
                word.pop_back();

            if (word.empty() || word.size() < 2) continue;

            // Skip stopwords
            if (std::find(stopwords.begin(), stopwords.end(), word) != stopwords.end())
                continue;

            // Deduplicate
            if (std::find(keywords.begin(), keywords.end(), word) == keywords.end())
                keywords.push_back(word);
        }

        return keywords;
    }

    std::string QueryAnalyzer::BuildFts5Query(const std::vector<std::string>& keywords) {
        if (keywords.empty()) return "*";

        std::string query;
        for (size_t i = 0; i < keywords.size(); ++i) {
            if (i > 0) query += " OR ";
            query += "\"" + keywords[i] + "\"";
        }
        return query;
    }

    std::string QueryAnalyzer::Analyze(const std::string& query) {
        auto keywords = ExtractKeywords(query);
        return BuildFts5Query(keywords);
    }

} // namespace LocalNotebookLLM::Application

