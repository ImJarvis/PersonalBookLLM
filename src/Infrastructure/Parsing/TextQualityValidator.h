#pragma once
#include <string>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief Validates extracted text quality to detect garbage/corrupt PDF extraction.
    /// Uses character-level entropy analysis and pattern detection.
    class TextQualityValidator {
    public:
        struct ValidationResult {
            bool   isValid       = true;
            float  qualityScore  = 1.0f;   // 0.0 (garbage) → 1.0 (clean)
            std::string warning;            // Empty if valid
        };

        /// Validate a text block for quality issues.
        [[nodiscard]]
        static ValidationResult Validate(const std::string& text);

        /// Check if a full document's text extraction has acceptable quality.
        /// Uses sampling — checks first, middle, and last pages.
        [[nodiscard]]
        static ValidationResult ValidateDocumentSample(
            const std::string& fullText, int pageCount);

    private:
        /// Character-level readability ratio (printable ASCII + known UTF-8 / total).
        [[nodiscard]]
        static float ComputeReadabilityRatio(const std::string& text);

        /// Word-level validity check (do most "words" look like real words?).
        [[nodiscard]]
        static float ComputeWordQuality(const std::string& text);

        /// Detection of common PDF extraction artifacts.
        [[nodiscard]]
        static bool HasExtractionArtifacts(const std::string& text);

        // ─── Thresholds ───
        static constexpr float MIN_READABILITY_RATIO = 0.60f;
        static constexpr float MIN_WORD_QUALITY      = 0.50f;
        static constexpr float WARNING_THRESHOLD     = 0.75f;
    };

} // namespace LocalNotebookLLM::Infrastructure
