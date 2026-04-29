#include "Infrastructure/Search/ReRanker.h"
#include "Core/Log.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>

namespace LocalNotebookLLM::Infrastructure {

    std::vector<std::string> ReRanker::Tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::string current;
        
        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                current += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            } else {
                if (!current.empty()) {
                    // Skip very short tokens (articles, prepositions)
                    if (current.size() > 2) {
                        tokens.push_back(std::move(current));
                    }
                    current.clear();
                }
            }
        }
        if (!current.empty() && current.size() > 2) {
            tokens.push_back(std::move(current));
        }
        return tokens;
    }

    float ReRanker::ComputeOverlap(const std::string& query, const std::string& passage) {
        auto queryTerms = Tokenize(query);
        if (queryTerms.empty()) return 0.0f;

        // Convert passage to lowercase for case-insensitive matching
        std::string lowerPassage;
        lowerPassage.reserve(passage.size());
        for (char c : passage) {
            lowerPassage += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        int hits = 0;
        for (const auto& term : queryTerms) {
            if (lowerPassage.find(term) != std::string::npos) {
                hits++;
            }
        }

        return static_cast<float>(hits) / static_cast<float>(queryTerms.size());
    }

    std::vector<Core::HybridSearchResult> ReRanker::ReRank(
        const std::string& query,
        std::vector<Core::HybridSearchResult>& results,
        float originalWeight,
        float overlapWeight)
    {
        if (results.empty()) return results;

        // Compute overlap score for each result
        struct ScoredResult {
            Core::HybridSearchResult result;
            float combinedScore;
        };

        std::vector<ScoredResult> scored;
        scored.reserve(results.size());

        // Normalize original confidence scores to [0, 1] range
        double maxConfidence = 0.0;
        for (const auto& r : results) {
            maxConfidence = std::max(maxConfidence, std::abs(r.confidenceScore));
        }
        if (maxConfidence == 0.0) maxConfidence = 1.0;

        for (auto& r : results) {
            float normalizedOriginal = static_cast<float>(std::abs(r.confidenceScore) / maxConfidence);
            float overlap = ComputeOverlap(query, r.content);
            float combined = originalWeight * normalizedOriginal + overlapWeight * overlap;
            
            // Update confidence score to the re-ranked score
            r.confidenceScore = static_cast<double>(combined);
            scored.push_back({std::move(r), combined});
        }

        // Sort by combined score descending
        std::sort(scored.begin(), scored.end(),
            [](const ScoredResult& a, const ScoredResult& b) {
                return a.combinedScore > b.combinedScore;
            });

        std::vector<Core::HybridSearchResult> reranked;
        reranked.reserve(scored.size());
        for (auto& s : scored) {
            reranked.push_back(std::move(s.result));
        }

        LOG_DEBUG("ReRanker", "Re-ranked " + std::to_string(reranked.size()) + " results");
        return reranked;
    }

} // namespace LocalNotebookLLM::Infrastructure
