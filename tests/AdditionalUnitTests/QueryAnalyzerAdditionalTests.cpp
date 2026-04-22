#include <gtest/gtest.h>
#include "Application/QueryAnalyzer.h"
#include <algorithm>

using namespace LocalNotebookLLM::Application;

TEST(QueryAnalyzerAdditionalTest, DetectIntentRecognizesSummationPhrases) {
    EXPECT_EQ(QueryAnalyzer::DetectIntent("Give me an overview of this document"),
              QueryIntent::Summation);
    EXPECT_EQ(QueryAnalyzer::DetectIntent("What is this about?"),
              QueryIntent::Summation);
}

TEST(QueryAnalyzerAdditionalTest, DetectIntentRecognizesChapterSummaryByWordOrdinal) {
    EXPECT_EQ(QueryAnalyzer::DetectIntent("Summarize chapter five"),
              QueryIntent::ChapterSummary);
}

TEST(QueryAnalyzerAdditionalTest, DetectIntentDefaultsToKeywordForFactQueries) {
    EXPECT_EQ(QueryAnalyzer::DetectIntent("What was the operating margin in Q3?"),
              QueryIntent::Keyword);
}

TEST(QueryAnalyzerAdditionalTest, ExtractKeywordsPreservesDomainTerms) {
    auto keywords = QueryAnalyzer::ExtractKeywords("Please summarize chapter 3 overview");

    EXPECT_TRUE(std::find(keywords.begin(), keywords.end(), "summarize") != keywords.end());
    EXPECT_TRUE(std::find(keywords.begin(), keywords.end(), "chapter") != keywords.end());
    EXPECT_TRUE(std::find(keywords.begin(), keywords.end(), "overview") != keywords.end());
}

TEST(QueryAnalyzerAdditionalTest, AnalyzeReturnsWildcardForStopwordOnlyInput) {
    auto query = QueryAnalyzer::Analyze("the and if or to in");
    EXPECT_EQ(query, "*");
}
