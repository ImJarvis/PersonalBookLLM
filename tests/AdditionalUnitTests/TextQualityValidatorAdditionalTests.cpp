#include <gtest/gtest.h>
#include "Infrastructure/Parsing/TextQualityValidator.h"

using namespace LocalNotebookLLM::Infrastructure;

TEST(TextQualityValidatorAdditionalTest, FlagsExcessiveWhitespaceArtifacts) {
    std::string text = "This paragraph is valid text.";
    text += std::string(40, ' ');
    text += "More text after an extraction artifact block.";

    auto result = TextQualityValidator::Validate(text);

    EXPECT_TRUE(result.isValid);
    EXPECT_FALSE(result.warning.empty());
    EXPECT_LT(result.qualityScore, 1.0f);
}

TEST(TextQualityValidatorAdditionalTest, ControlCharacterHeavyTextFails) {
    std::string text = "Readable prefix ";
    for (int i = 0; i < 20; ++i) {
        text.push_back('\x02');
    }
    text += " suffix";

    auto result = TextQualityValidator::Validate(text);

    EXPECT_FALSE(result.isValid);
    EXPECT_FALSE(result.warning.empty());
}

TEST(TextQualityValidatorAdditionalTest, NumericHeavyButReadableTextPasses) {
    auto result = TextQualityValidator::Validate(
        "2025 11 42 300 19 7 88 1234 5678 90 with enough words for parser quality");

    EXPECT_TRUE(result.isValid);
    EXPECT_GE(result.qualityScore, 0.5f);
}

TEST(TextQualityValidatorAdditionalTest, LongDocumentSampleValidationWorks) {
    std::string chunk =
        "This is a clean sentence used for long document sampling validation and readability checks. ";
    std::string longText;
    for (int i = 0; i < 120; ++i) {
        longText += chunk;
    }

    auto result = TextQualityValidator::ValidateDocumentSample(longText, 20);

    EXPECT_TRUE(result.isValid);
    EXPECT_GE(result.qualityScore, 0.7f);
}
