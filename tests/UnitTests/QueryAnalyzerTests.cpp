#include <gtest/gtest.h>
#include "Application/QueryAnalyzer.h"

using namespace LocalNotebookLLM::Application;

TEST(QueryAnalyzerTest, ExtractsKeywordsFromSimpleQuery) {
    auto keywords = QueryAnalyzer::ExtractKeywords("What was the total revenue in Q3?");

    // Should contain meaningful terms, not stopwords
    EXPECT_TRUE(std::find(keywords.begin(), keywords.end(), "total") != keywords.end());
    EXPECT_TRUE(std::find(keywords.begin(), keywords.end(), "revenue") != keywords.end());
    EXPECT_TRUE(std::find(keywords.begin(), keywords.end(), "q3") != keywords.end());

    // Should NOT contain stopwords
    EXPECT_TRUE(std::find(keywords.begin(), keywords.end(), "what") == keywords.end());
    EXPECT_TRUE(std::find(keywords.begin(), keywords.end(), "was") == keywords.end());
    EXPECT_TRUE(std::find(keywords.begin(), keywords.end(), "the") == keywords.end());
    EXPECT_TRUE(std::find(keywords.begin(), keywords.end(), "in") == keywords.end());
}

TEST(QueryAnalyzerTest, HandlesEmptyQuery) {
    auto keywords = QueryAnalyzer::ExtractKeywords("");
    EXPECT_TRUE(keywords.empty());
}

TEST(QueryAnalyzerTest, HandlesSingleWordQuery) {
    auto keywords = QueryAnalyzer::ExtractKeywords("revenue");
    ASSERT_EQ(keywords.size(), 1);
    EXPECT_EQ(keywords[0], "revenue");
}

TEST(QueryAnalyzerTest, DeduplicatesKeywords) {
    auto keywords = QueryAnalyzer::ExtractKeywords("revenue revenue revenue");
    ASSERT_EQ(keywords.size(), 1);
    EXPECT_EQ(keywords[0], "revenue");
}

TEST(QueryAnalyzerTest, StripsPunctuation) {
    auto keywords = QueryAnalyzer::ExtractKeywords("What about \"margins\"?");
    // "margins" should appear without quotes
    bool found = std::find(keywords.begin(), keywords.end(), "margins") != keywords.end();
    EXPECT_TRUE(found);
}

TEST(QueryAnalyzerTest, SkipsShortWords) {
    auto keywords = QueryAnalyzer::ExtractKeywords("a b c revenue");
    ASSERT_EQ(keywords.size(), 1);
    EXPECT_EQ(keywords[0], "revenue");
}

TEST(QueryAnalyzerTest, BuildsFts5Query) {
    std::vector<std::string> keywords = {"revenue", "q3", "total"};
    auto ftsQuery = QueryAnalyzer::BuildFts5Query(keywords);

    // Should contain quoted terms joined with OR
    EXPECT_TRUE(ftsQuery.find("\"revenue\"") != std::string::npos);
    EXPECT_TRUE(ftsQuery.find("\"q3\"") != std::string::npos);
    EXPECT_TRUE(ftsQuery.find("OR") != std::string::npos);
}

TEST(QueryAnalyzerTest, EmptyKeywordsProduceWildcard) {
    auto ftsQuery = QueryAnalyzer::BuildFts5Query({});
    EXPECT_EQ(ftsQuery, "*");
}

TEST(QueryAnalyzerTest, AnalyzeEndToEnd) {
    auto ftsQuery = QueryAnalyzer::Analyze("What risk factors were identified?");
    EXPECT_FALSE(ftsQuery.empty());
    EXPECT_TRUE(ftsQuery.find("\"risk\"") != std::string::npos);
    EXPECT_TRUE(ftsQuery.find("\"factors\"") != std::string::npos);
    EXPECT_TRUE(ftsQuery.find("\"identified\"") != std::string::npos);
}
