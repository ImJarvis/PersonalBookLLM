#include <gtest/gtest.h>
#include "Infrastructure/Parsing/TextQualityValidator.h"

using namespace LocalNotebookLLM::Infrastructure;

TEST(TextQualityValidatorTest, CleanTextPassesValidation) {
    auto result = TextQualityValidator::Validate(
        "This is clean, readable English text with proper formatting. "
        "It contains multiple sentences and should score highly.");

    EXPECT_TRUE(result.isValid);
    EXPECT_GE(result.qualityScore, 0.7f);
    EXPECT_TRUE(result.warning.empty());
}

TEST(TextQualityValidatorTest, EmptyTextFailsValidation) {
    auto result = TextQualityValidator::Validate("");
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.qualityScore, 0.0f);
}

TEST(TextQualityValidatorTest, GarbageTextFailsValidation) {
    // Simulate garbled PDF extraction with non-printable chars
    std::string garbage(100, '\x01');
    auto result = TextQualityValidator::Validate(garbage);
    EXPECT_FALSE(result.isValid);
    EXPECT_LT(result.qualityScore, 0.5f);
}

TEST(TextQualityValidatorTest, MixedQualityGeneratesWarning) {
    // Mix of readable and non-readable chars
    std::string mixed = "This is readable text. ";
    mixed += std::string(50, '\x80');  // Non-printable
    mixed += " More readable text here.";

    auto result = TextQualityValidator::Validate(mixed);
    // Should be valid but with lower quality score
    EXPECT_LE(result.qualityScore, 0.9f);
}

TEST(TextQualityValidatorTest, DocumentSampleValidation) {
    std::string fullText =
        "Page 1: Introduction to the report. This section covers the main objectives. "
        "Page 2: Methodology used in the study. We employed statistical analysis. "
        "Page 3: Results show significant improvements across all metrics. ";

    auto result = TextQualityValidator::ValidateDocumentSample(fullText, 3);
    EXPECT_TRUE(result.isValid);
    EXPECT_GE(result.qualityScore, 0.7f);
}
