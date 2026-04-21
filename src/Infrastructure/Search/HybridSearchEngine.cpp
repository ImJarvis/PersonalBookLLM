#include "Infrastructure/Search/HybridSearchEngine.h"
#include "Application/QueryAnalyzer.h"
#include "Infrastructure/Indexing/DatabaseManager.h"
#include <sqlite3.h>
#include "Core/Log.h"
#include <algorithm>
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

                // Load all structural embeddings into memory and compute cosine similarity
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
                            // Compute Cosine Similarity (vectors should already be L2 normalized)
                            float score = 0.0f;
                            for (size_t i = 0; i < dim; ++i) {
                                score += queryVec[i] * blob[i];
                            }
                            
                            // High similarity cutoff -> 0.4
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
                
                // Sort by highest cosine similarity
                std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                    return a.score > b.score;
                });
                
                // Keep Top-K
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
                    hr.resolvedTier = Core::SearchTier::Structural; // Generic enum mapping
                    semanticResults.push_back(std::move(hr));
                }
                LOG_INFO("Search", "Tier 0 (Semantic): " + std::to_string(semanticResults.size()) + " matches > 0.4");
            }
        }

        // Merge semantic results into the existing merged results
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

} // namespace LocalNotebookLLM::Infrastructure
