#include "Infrastructure/Search/HybridSearchEngine.h"
#include "Application/QueryAnalyzer.h"
#include "Infrastructure/Indexing/DatabaseManager.h"
#include <sqlite3.h>
#include "Core/Log.h"
#include <algorithm>
#include <cctype>
#include <regex>
#include <set>

namespace LocalNotebookLLM::Infrastructure {

    HybridSearchEngine::HybridSearchEngine(
        std::shared_ptr<Core::IStructuralIndexer> fts5Indexer,
        std::shared_ptr<StructuralNavigator> navigator,
        std::shared_ptr<Core::IEmbeddingProvider> embeddingProvider,
        std::shared_ptr<DatabaseManager> db)
        : m_fts5Indexer(std::move(fts5Indexer))
        , m_navigator(std::move(navigator))
        , m_embeddingProvider(std::move(embeddingProvider))
        , m_db(std::move(db))
    {}

    std::vector<Core::HybridSearchResult>
    HybridSearchEngine::Search(const std::string& query, const Core::HybridSearchOptions& opts) {
        LOG_INFO("Search", "Search() query: \"" + query + "\" topK=" + std::to_string(opts.topK));

        // ═══════════════════ INTENT DETECTION ═══════════════════
        // Classify the query before doing anything else.
        // Summary/overview queries need a completely different fetch strategy:
        //   - BM25 keyword search returns NOTHING for "what is this about" because
        //     every content word is a stopword in FTS5 terms.
        //   - The right strategy is to fetch the document's structural skeleton.
        auto intent = Application::QueryAnalyzer::DetectIntent(query);
        LOG_INFO("Search", "Intent: " +
            std::string(intent == Application::QueryIntent::Summation      ? "Summation" :
                        intent == Application::QueryIntent::ChapterSummary ? "ChapterSummary" :
                                                                             "Keyword"));

        if (intent == Application::QueryIntent::Summation) {
            // Full-document overview: fetch title, all headings, and opening paragraphs
            auto overviewResults = GetDocumentOverview(opts.documentIdFilter, opts.topK);
            LOG_INFO("Search", "Summation path: " + std::to_string(overviewResults.size()) + " overview nodes");
            if (!overviewResults.empty()) return overviewResults;
            // Fall through to keyword search if no structural data available
        }

        if (intent == Application::QueryIntent::ChapterSummary) {
            // Chapter-specific summary: extract the chapter identifier and fetch its subtree
            auto chapterResults = GetChapterContent(query, opts.documentIdFilter, opts.topK);
            LOG_INFO("Search", "ChapterSummary path: " + std::to_string(chapterResults.size()) + " chapter nodes");
            if (!chapterResults.empty()) return chapterResults;
            // Fall through to structural navigation if no exact chapter match
        }

        // Prepare search query via QueryAnalyzer
        std::string ftsQuery = Application::QueryAnalyzer::Analyze(query);
        LOG_DEBUG("Search", "FTS query: \"" + ftsQuery + "\"");

        // ═══════════════════ TIER 1: BM25 Keyword Search ═══════════════════
        Core::SearchOptions searchOpts;
        searchOpts.topK = opts.topK * 2;  // Fetch extra for filtering
        searchOpts.documentIdFilter = opts.documentIdFilter;

        auto bm25Results = m_fts5Indexer->Search(ftsQuery, searchOpts);
        LOG_INFO("Search", "Tier 1 (BM25): " + std::to_string(bm25Results.size()) + " results");

        if (!bm25Results.empty()) {
            double topScore = bm25Results[0].bm25Score;
            LOG_DEBUG("Search", "  Top BM25 score: " + std::to_string(topScore) +
                " (threshold: " + std::to_string(opts.bm25ConfidenceThreshold) + ")");
            if (topScore >= opts.bm25ConfidenceThreshold) {
                LOG_INFO("Search", "  BM25 confident — returning Tier 1 results");
                return ConvertToHybrid(bm25Results, Core::SearchTier::BM25, opts.topK);
            }
        }

        // ═══════════════════ TIER 2: Structural Tree Navigation ═══════════════════
        LOG_INFO("Search", "Tier 2 (Structural): navigating...");
        auto structuralResults = m_navigator->Navigate(query, opts.topK);
        LOG_INFO("Search", "Tier 2: " + std::to_string(structuralResults.size()) + " results");

        // Merge BM25 + structural results
        auto bm25Hybrid = ConvertToHybrid(bm25Results, Core::SearchTier::BM25, opts.topK * 2);
        auto structHybrid = ConvertToHybrid(structuralResults, Core::SearchTier::Structural, opts.topK * 2);
        auto merged = MergeAndDeduplicate(bm25Hybrid, structHybrid);
        LOG_INFO("Search", "Merged: " + std::to_string(merged.size()) + " results");

        if (!merged.empty()) {
            double topConfidence = merged[0].confidenceScore;
            if (topConfidence >= opts.structuralConfidenceThreshold) {
                LOG_INFO("Search", "  Structural confident — returning merged results");
                return TakeTop(merged, opts.topK);
            }
        }

        // ═══════════════════ TIER 0: Semantic Embedding Matching ═══════════════════
        std::vector<Core::HybridSearchResult> semanticResults;
        if (opts.enableEmbeddingFallback && m_embeddingProvider && m_embeddingProvider->IsLoaded() && m_db) {
            LOG_INFO("Search", "Tier 0 (Semantic): Embedding query...");
            auto queryEmbResult = m_embeddingProvider->Embed(query);
            if (queryEmbResult) {
                const std::vector<float>& queryVec = *queryEmbResult;

                std::string sql = R"(
                    SELECT id, document_id, node_type, page_number, section_path, content, embedding
                    FROM document_nodes
                    WHERE embedding IS NOT NULL
                )";
                if (opts.documentIdFilter >= 0) {
                    sql += " AND document_id = " + std::to_string(opts.documentIdFilter);
                }

                auto stmt = m_db->Prepare(sql);
                
                struct ScoredSemantic {
                    Core::SearchResult res;
                    float score;
                };
                std::vector<ScoredSemantic> candidates;

                while (stmt.Step()) {
                    int bytes = sqlite3_column_bytes(stmt.GetHandle(), 6);
                    if (bytes > 0 && bytes % sizeof(float) == 0) {
                        const float* blob = reinterpret_cast<const float*>(sqlite3_column_blob(stmt.GetHandle(), 6));
                        size_t dim = bytes / sizeof(float);
                        
                        if (dim == queryVec.size()) {
                            float score = 0.0f;
                            for (size_t i = 0; i < dim; ++i) {
                                score += queryVec[i] * blob[i];
                            }
                            
                            if (score > 0.4f) {
                                Core::SearchResult r;
                                r.nodeId = stmt.GetInt64(0);
                                r.documentId = stmt.GetInt64(1);
                                r.nodeType = Core::StringToNodeType(stmt.GetString(2));
                                r.pageNumber = stmt.GetInt(3);
                                r.sectionPath = stmt.GetString(4);
                                r.content = stmt.GetString(5);
                                r.bm25Score = score;
                                candidates.push_back({std::move(r), score});
                            }
                        }
                    }
                }
                
                std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                    return a.score > b.score;
                });
                
                for (size_t i = 0; i < candidates.size() && i < static_cast<size_t>(opts.topK); ++i) {
                    Core::HybridSearchResult hr;
                    hr.nodeId = candidates[i].res.nodeId;
                    hr.documentId = candidates[i].res.documentId;
                    hr.nodeType = candidates[i].res.nodeType;
                    hr.pageNumber = candidates[i].res.pageNumber;
                    hr.sectionPath = candidates[i].res.sectionPath;
                    hr.content = candidates[i].res.content;
                    hr.bm25Score = candidates[i].score;
                    hr.confidenceScore = candidates[i].score;
                    hr.resolvedTier = Core::SearchTier::Structural;
                    semanticResults.push_back(std::move(hr));
                }
                LOG_INFO("Search", "Tier 0 (Semantic): " + std::to_string(semanticResults.size()) + " matches > 0.4");
            }
        }

        merged = MergeAndDeduplicate(merged, semanticResults);

        LOG_INFO("Search", "Returning " + std::to_string(merged.size()) + " final results");
        return TakeTop(merged, opts.topK);
    }


    std::vector<Core::HybridSearchResult>
    HybridSearchEngine::ConvertToHybrid(
        const std::vector<Core::SearchResult>& results,
        Core::SearchTier tier, int limit)
    {
        std::vector<Core::HybridSearchResult> hybrid;
        int count = 0;
        for (const auto& r : results) {
            if (count >= limit) break;

            Core::HybridSearchResult hr;
            // Copy base fields
            hr.nodeId      = r.nodeId;
            hr.documentId  = r.documentId;
            hr.nodeType    = r.nodeType;
            hr.pageNumber  = r.pageNumber;
            hr.sectionPath = r.sectionPath;
            hr.content     = r.content;
            hr.bm25Score   = r.bm25Score;
            // Set hybrid-specific fields
            hr.resolvedTier    = tier;
            hr.confidenceScore = r.bm25Score;  // Normalize later if needed
            hybrid.push_back(std::move(hr));
            count++;
        }
        return hybrid;
    }

    std::vector<Core::HybridSearchResult>
    HybridSearchEngine::MergeAndDeduplicate(
        const std::vector<Core::HybridSearchResult>& a,
        const std::vector<Core::HybridSearchResult>& b)
    {
        std::vector<Core::HybridSearchResult> merged;
        std::set<int64_t> seenIds;

        // Add all from 'a' first
        for (const auto& r : a) {
            if (seenIds.insert(r.nodeId).second) {
                merged.push_back(r);
            }
        }

        // Add from 'b' only if not already present (or if higher score)
        for (const auto& r : b) {
            if (seenIds.insert(r.nodeId).second) {
                merged.push_back(r);
            }
        }

        // Sort by confidence score descending
        std::sort(merged.begin(), merged.end(),
                  [](const auto& x, const auto& y) {
                      return x.confidenceScore > y.confidenceScore;
                  });

        return merged;
    }

    std::vector<Core::HybridSearchResult>
    HybridSearchEngine::TakeTop(std::vector<Core::HybridSearchResult>& results, int n) {
        if (static_cast<int>(results.size()) <= n) return results;
        results.resize(static_cast<size_t>(n));
        return results;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  GetDocumentOverview — Summation intent handler
    //
    //  Fetches the document's structural skeleton in reading order:
    //  - All title/h1/h2/h3 headings (the table-of-contents)
    //  - The first non-empty paragraph under each heading
    //
    //  This gives the model enough context to answer "what is this about"
    //  or "summarize this document" without needing keyword BM25 matches.
    // ─────────────────────────────────────────────────────────────────────
    std::vector<Core::HybridSearchResult>
    HybridSearchEngine::GetDocumentOverview(int64_t documentIdFilter, int topK) const {
        if (!m_db) return {};

        std::string docFilter = documentIdFilter >= 0
            ? " AND document_id = " + std::to_string(documentIdFilter)
            : "";

        // Step 1: Fetch all headings in reading order
        std::string headingSql = R"(
            SELECT id, document_id, node_type, page_number, section_path, content, sort_order
            FROM document_nodes
            WHERE node_type IN ('title', 'h1', 'h2', 'h3')
        )" + docFilter + R"(
            ORDER BY document_id, sort_order
            LIMIT 50
        )";

        auto headingStmt = m_db->Prepare(headingSql);

        struct HeadingInfo {
            int64_t id;
            int64_t documentId;
            Core::NodeType nodeType;
            int pageNumber;
            std::string sectionPath;
            std::string content;
            int sortOrder;
        };
        std::vector<HeadingInfo> headings;
        while (headingStmt.Step()) {
            headings.push_back({
                headingStmt.GetInt64(0),
                headingStmt.GetInt64(1),
                Core::StringToNodeType(headingStmt.GetString(2)),
                headingStmt.GetInt(3),
                headingStmt.GetString(4),
                headingStmt.GetString(5),
                headingStmt.GetInt(6)
            });
        }

        if (headings.empty()) {
            // No headings — fall back to fetching the first N paragraphs in order
            std::string parasSql = R"(
                SELECT id, document_id, node_type, page_number, section_path, content
                FROM document_nodes
                WHERE node_type = 'paragraph'
            )" + docFilter + R"(
                ORDER BY document_id, sort_order
                LIMIT )" + std::to_string(topK);

            auto parasStmt = m_db->Prepare(parasSql);
            std::vector<Core::HybridSearchResult> fallback;
            while (parasStmt.Step()) {
                Core::HybridSearchResult hr;
                hr.nodeId         = parasStmt.GetInt64(0);
                hr.documentId     = parasStmt.GetInt64(1);
                hr.nodeType       = Core::StringToNodeType(parasStmt.GetString(2));
                hr.pageNumber     = parasStmt.GetInt(3);
                hr.sectionPath    = parasStmt.GetString(4);
                hr.content        = parasStmt.GetString(5);
                hr.bm25Score      = 1.0;
                hr.confidenceScore = 1.0;
                hr.resolvedTier   = Core::SearchTier::Structural;
                fallback.push_back(std::move(hr));
            }
            return fallback;
        }

        std::vector<Core::HybridSearchResult> results;
        results.reserve(headings.size() * 2);

        for (const auto& h : headings) {
            // Add the heading itself
            Core::HybridSearchResult hr;
            hr.nodeId          = h.id;
            hr.documentId      = h.documentId;
            hr.nodeType        = h.nodeType;
            hr.pageNumber      = h.pageNumber;
            hr.sectionPath     = h.sectionPath;
            hr.content         = h.content;
            hr.bm25Score       = 1.0;
            hr.confidenceScore = 1.0;
            hr.resolvedTier    = Core::SearchTier::Structural;
            results.push_back(std::move(hr));

            // Add the first paragraph child under this heading
            auto childStmt = m_db->Prepare(R"(
                SELECT id, document_id, node_type, page_number, section_path, content
                FROM document_nodes
                WHERE parent_id = ? AND node_type = 'paragraph' AND content != ''
                ORDER BY sort_order
                LIMIT 1
            )");
            childStmt.Bind(1, h.id);
            if (childStmt.Step()) {
                Core::HybridSearchResult childHr;
                childHr.nodeId          = childStmt.GetInt64(0);
                childHr.documentId      = childStmt.GetInt64(1);
                childHr.nodeType        = Core::StringToNodeType(childStmt.GetString(2));
                childHr.pageNumber      = childStmt.GetInt(3);
                childHr.sectionPath     = childStmt.GetString(4);
                childHr.content         = childStmt.GetString(5);
                childHr.bm25Score       = 0.9;
                childHr.confidenceScore = 0.9;
                childHr.resolvedTier    = Core::SearchTier::Structural;
                results.push_back(std::move(childHr));
            }

            if (static_cast<int>(results.size()) >= topK) break;
        }

        return results;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  GetChapterContent — ChapterSummary intent handler
    //
    //  Finds a heading that references the requested chapter (by number or
    //  ordinal word) and fetches all content paragraphs beneath it.
    // ─────────────────────────────────────────────────────────────────────
    std::vector<Core::HybridSearchResult>
    HybridSearchEngine::GetChapterContent(
        const std::string& query, int64_t documentIdFilter, int topK) const
    {
        if (!m_db) return {};

        // Extract chapter number/identifier from the query
        // Matches: "chapter 1", "chapter one", "ch. 2", "section 3", "part II"
        std::string lower = query;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        // Map ordinal words to digits
        static const std::vector<std::pair<std::string, std::string>> ordinals = {
            {"one","1"}, {"two","2"}, {"three","3"}, {"four","4"}, {"five","5"},
            {"six","6"}, {"seven","7"}, {"eight","8"}, {"nine","9"}, {"ten","10"},
        };
        for (const auto& [word, digit] : ordinals) {
            size_t pos = lower.find(word);
            if (pos != std::string::npos) {
                lower.replace(pos, word.size(), digit);
            }
        }

        // Extract the chapter number (digits after chapter/section/part)
        std::regex numPattern(R"((?:chapter|section|part|ch\.?)\s*(\d+))", std::regex::icase);
        std::smatch m;
        std::string chNum;
        if (std::regex_search(lower, m, numPattern)) {
            chNum = m[1].str();
        }

        if (chNum.empty()) return {};  // Can't identify chapter — fall through

        std::string docFilter = documentIdFilter >= 0
            ? " AND document_id = " + std::to_string(documentIdFilter)
            : "";

        // Find headings containing the chapter number
        // We look for "chapter N", "N.", or just the number in the heading text
        std::string headingSql = R"(
            SELECT id, document_id, node_type, page_number, section_path, content
            FROM document_nodes
            WHERE node_type IN ('title', 'h1', 'h2', 'h3')
            AND (
                LOWER(content) LIKE '%chapter )" + chNum + R"(%'
                OR LOWER(content) LIKE '%)" + chNum + R"(.%'
                OR LOWER(content) LIKE '% )" + chNum + R"( %'
            )
        )" + docFilter + R"(
            ORDER BY document_id, sort_order
            LIMIT 3
        )";

        auto headingStmt = m_db->Prepare(headingSql);
        std::vector<Core::HybridSearchResult> results;

        while (headingStmt.Step()) {
            int64_t headingId = headingStmt.GetInt64(0);

            // Add the heading
            Core::HybridSearchResult hr;
            hr.nodeId          = headingId;
            hr.documentId      = headingStmt.GetInt64(1);
            hr.nodeType        = Core::StringToNodeType(headingStmt.GetString(2));
            hr.pageNumber      = headingStmt.GetInt(3);
            hr.sectionPath     = headingStmt.GetString(4);
            hr.content         = headingStmt.GetString(5);
            hr.bm25Score       = 1.0;
            hr.confidenceScore = 1.0;
            hr.resolvedTier    = Core::SearchTier::Structural;
            results.push_back(std::move(hr));

            // Fetch all paragraph children of this chapter heading
            auto childStmt = m_db->Prepare(R"(
                SELECT id, document_id, node_type, page_number, section_path, content
                FROM document_nodes
                WHERE parent_id = ? AND content != ''
                ORDER BY sort_order
                LIMIT 20
            )");
            childStmt.Bind(1, headingId);
            while (childStmt.Step() && static_cast<int>(results.size()) < topK) {
                Core::HybridSearchResult childHr;
                childHr.nodeId          = childStmt.GetInt64(0);
                childHr.documentId      = childStmt.GetInt64(1);
                childHr.nodeType        = Core::StringToNodeType(childStmt.GetString(2));
                childHr.pageNumber      = childStmt.GetInt(3);
                childHr.sectionPath     = childStmt.GetString(4);
                childHr.content         = childStmt.GetString(5);
                childHr.bm25Score       = 0.95;
                childHr.confidenceScore = 0.95;
                childHr.resolvedTier    = Core::SearchTier::Structural;
                results.push_back(std::move(childHr));
            }
        }

        return results;
    }

} // namespace LocalNotebookLLM::Infrastructure

