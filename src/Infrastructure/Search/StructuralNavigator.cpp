#include "Infrastructure/Search/StructuralNavigator.h"
#include "Infrastructure/Indexing/DatabaseManager.h"
#include "Application/QueryAnalyzer.h"
#include "Core/Enums/NodeType.h"
#include <algorithm>
#include <sstream>
#include <cctype>

namespace LocalNotebookLLM::Infrastructure {

    StructuralNavigator::StructuralNavigator(std::shared_ptr<DatabaseManager> db)
        : m_db(std::move(db)) {}

    std::vector<Core::SearchResult>
    StructuralNavigator::Navigate(const std::string& query, int topK) const {
        auto keywords = Application::QueryAnalyzer::ExtractKeywords(query);
        if (keywords.empty()) return {};

        // Query all heading nodes
        auto stmt = m_db->Prepare(R"(
            SELECT id, document_id, node_type, page_number, section_path, content
            FROM document_nodes
            WHERE node_type IN ('title', 'h1', 'h2', 'h3')
            ORDER BY document_id, sort_order
        )");

        // Score each heading
        struct ScoredHeading {
            Core::SearchResult result;
            double score;
        };
        std::vector<ScoredHeading> candidates;

        while (stmt.Step()) {
            Core::SearchResult r;
            r.nodeId      = stmt.GetInt64(0);
            r.documentId  = stmt.GetInt64(1);
            r.nodeType    = Core::StringToNodeType(stmt.GetString(2));
            r.pageNumber  = stmt.GetInt(3);
            r.sectionPath = stmt.GetString(4);
            r.content     = stmt.GetString(5);

            // Score: keyword overlap in heading text + section path
            double score = ScoreHeadingRelevance(r.content, keywords);
            score += ScoreHeadingRelevance(r.sectionPath, keywords) * 0.5;

            // Heading-Boost: H1/H2 headings get multiplied score for broader relevance
            if (r.nodeType == Core::NodeType::H1) score *= 1.2;
            if (r.nodeType == Core::NodeType::H2) score *= 1.1;

            if (score > 0.0) {
                r.bm25Score = score;
                candidates.push_back({std::move(r), score});
            }
        }

        // Sort by score descending
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.score > b.score; });

        // Take top-K headings and collect their child content
        std::vector<Core::SearchResult> results;
        int count = 0;
        for (const auto& candidate : candidates) {
            if (count >= topK) break;

            // Add the heading itself
            results.push_back(candidate.result);
            count++;

            // Fetch child content under this heading
            auto childStmt = m_db->Prepare(R"(
                SELECT id, document_id, node_type, page_number, section_path, content
                FROM document_nodes
                WHERE parent_id = ?
                ORDER BY sort_order
                LIMIT 5
            )");
            childStmt.Bind(1, candidate.result.nodeId);

            while (childStmt.Step() && count < topK * 2) {
                Core::SearchResult child;
                child.nodeId      = childStmt.GetInt64(0);
                child.documentId  = childStmt.GetInt64(1);
                child.nodeType    = Core::StringToNodeType(childStmt.GetString(2));
                child.pageNumber  = childStmt.GetInt(3);
                child.sectionPath = childStmt.GetString(4);
                child.content     = childStmt.GetString(5);
                child.bm25Score   = candidate.score * 0.8;  // Inherit parent's relevance
                results.push_back(std::move(child));
            }
        }

        return results;
    }

    double StructuralNavigator::ScoreHeadingRelevance(
        const std::string& headingText,
        const std::vector<std::string>& keywords)
    {
        if (headingText.empty() || keywords.empty()) return 0.0;

        // Lowercase the heading
        std::string lower = headingText;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        int matches = 0;
        for (const auto& kw : keywords) {
            if (lower.find(kw) != std::string::npos) {
                matches++;
            }
        }

        // Normalize: fraction of keywords found in this heading
        return static_cast<double>(matches) / static_cast<double>(keywords.size());
    }

} // namespace LocalNotebookLLM::Infrastructure
