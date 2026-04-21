#include "Infrastructure/Parsing/TextQualityValidator.h"
#include <algorithm>
#include <sstream>
#include <cctype>
#include <cmath>

namespace LocalNotebookLLM::Infrastructure {

    TextQualityValidator::ValidationResult
    TextQualityValidator::Validate(const std::string& text) {
        ValidationResult result;

        if (text.empty()) {
            result.isValid      = false;
            result.qualityScore = 0.0f;
            result.warning      = "Empty text content";
            return result;
        }

        float readability = ComputeReadabilityRatio(text);
        float wordQuality = ComputeWordQuality(text);
        bool  artifacts   = HasExtractionArtifacts(text);

        // Combined quality score (weighted)
        result.qualityScore = readability * 0.5f + wordQuality * 0.4f +
                              (artifacts ? 0.0f : 0.1f);

        // Determine validity
        if (readability < MIN_READABILITY_RATIO) {
            result.isValid = false;
            result.warning = "Text readability below threshold (" +
                             std::to_string(static_cast<int>(readability * 100)) + "%)";
        } else if (wordQuality < MIN_WORD_QUALITY) {
            result.isValid = false;
            result.warning = "Word quality too low — possible encoding corruption";
        } else if (result.qualityScore < WARNING_THRESHOLD) {
            result.isValid = true;
            result.warning = "Low extraction quality — some content may be garbled";
        }

        return result;
    }

    TextQualityValidator::ValidationResult
    TextQualityValidator::ValidateDocumentSample(const std::string& fullText, int pageCount) {
        // For short documents, validate everything
        if (fullText.size() < 5000 || pageCount <= 3) {
            return Validate(fullText);
        }

        // Sample: first 2000 chars, middle 2000, last 2000
        size_t len = fullText.size();
        std::string sample;
        sample += fullText.substr(0, 2000);
        sample += fullText.substr(len / 2 - 1000, 2000);
        sample += fullText.substr(len > 2000 ? len - 2000 : 0);

        return Validate(sample);
    }

    float TextQualityValidator::ComputeReadabilityRatio(const std::string& text) {
        if (text.empty()) return 0.0f;

        int readable = 0;
        int total = 0;

        for (unsigned char c : text) {
            total++;
            // Printable ASCII
            if (c >= 32 && c <= 126) { readable++; continue; }
            // Common whitespace
            if (c == '\n' || c == '\r' || c == '\t') { readable++; continue; }
            // Valid UTF-8 continuation bytes (0x80-0xBF following a lead byte is fine)
            if (c >= 0xC0 && c <= 0xF4) { readable++; continue; }  // UTF-8 lead bytes
            if (c >= 0x80 && c <= 0xBF) { readable++; continue; }  // UTF-8 continuation
        }

        return static_cast<float>(readable) / static_cast<float>(total);
    }

    float TextQualityValidator::ComputeWordQuality(const std::string& text) {
        std::istringstream stream(text);
        std::string word;
        int totalWords = 0;
        int validWords = 0;

        while (stream >> word && totalWords < 200) {  // Sample first 200 words
            totalWords++;

            // A "valid word" has mostly alphabetic characters
            int alpha = 0;
            for (unsigned char c : word) {
                if (std::isalpha(c)) alpha++;
            }

            if (word.empty()) continue;
            float alphaRatio = static_cast<float>(alpha) / static_cast<float>(word.size());

            // Words that are mostly letters, or are numbers/punctuation
            if (alphaRatio >= 0.5f || word.size() <= 2) {
                validWords++;
            }
        }

        if (totalWords == 0) return 0.0f;
        return static_cast<float>(validWords) / static_cast<float>(totalWords);
    }

    bool TextQualityValidator::HasExtractionArtifacts(const std::string& text) {
        // Common PDF extraction artifacts:
        // 1. Excessive whitespace (broken layout)
        int consecutiveSpaces = 0;
        int maxConsecutive = 0;
        for (char c : text) {
            if (c == ' ') {
                consecutiveSpaces++;
                maxConsecutive = std::max(maxConsecutive, consecutiveSpaces);
            } else {
                consecutiveSpaces = 0;
            }
        }
        if (maxConsecutive > 20) return true;

        // 2. Repeated control characters
        int controlCount = 0;
        for (unsigned char c : text) {
            if (c < 32 && c != '\n' && c != '\r' && c != '\t') {
                controlCount++;
            }
        }
        if (controlCount > 10) return true;

        // 3. Excessive single-character lines (typically from column extraction failures)
        std::istringstream stream(text);
        std::string line;
        int singleCharLines = 0;
        int totalLines = 0;
        while (std::getline(stream, line) && totalLines < 100) {
            totalLines++;
            if (line.size() == 1) singleCharLines++;
        }
        if (totalLines > 10 && singleCharLines > totalLines / 3) return true;

        return false;
    }

} // namespace LocalNotebookLLM::Infrastructure
