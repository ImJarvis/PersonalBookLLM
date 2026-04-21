#include <gtest/gtest.h>
#include "Application/ContextAssembler.h"
#include "Core/Models/SearchResult.h"
#include "Core/Enums/NodeType.h"

using namespace LocalNotebookLLM;

class ContextAssemblerTest : public ::testing::Test {
protected:
    Application::ContextAssembler assembler;

    static Core::SearchResult MakeResult(int page, const std::string& section,
                                          const std::string& content, double score = 1.0) {
        Core::SearchResult r;
        r.nodeId      = page * 100;
        r.documentId  = 1;
        r.nodeType    = Core::NodeType::Paragraph;
        r.pageNumber  = page;
        r.sectionPath = section;
        r.content     = content;
        r.bm25Score   = score;
        return r;
    }
};

TEST_F(ContextAssemblerTest, AssemblesBasicContext) {
    std::vector<Core::SearchResult> results = {
        MakeResult(1, "Introduction", "This is the introduction text."),
        MakeResult(5, "Methods", "We used statistical analysis."),
    };

    auto ctx = assembler.Assemble(results, "What methods were used?", 4096);

    EXPECT_FALSE(ctx.formattedContext.empty());
    EXPECT_EQ(ctx.sectionsIncluded, 2);
    EXPECT_EQ(ctx.sectionsOmitted, 0);
    EXPECT_EQ(ctx.sourceMap.size(), 2);
}

TEST_F(ContextAssemblerTest, IncludesSourceTags) {
    std::vector<Core::SearchResult> results = {
        MakeResult(12, "Financial Results > Q3", "Revenue was $4.2M"),
    };

    auto ctx = assembler.Assemble(results, "revenue", 4096);

    // Should contain the [SOURCE] tag with page and section
    EXPECT_TRUE(ctx.formattedContext.find("[SOURCE: Page 12") != std::string::npos);
    EXPECT_TRUE(ctx.formattedContext.find("Financial Results > Q3") != std::string::npos);
}

TEST_F(ContextAssemblerTest, RespectsTokenBudget) {
    // Create results that exceed the budget
    std::vector<Core::SearchResult> results;
    for (int i = 0; i < 50; ++i) {
        results.push_back(MakeResult(
            i, "Section " + std::to_string(i),
            std::string(500, 'x')  // 500 chars per result ≈ 125 tokens
        ));
    }

    // With a tiny budget, only some should fit
    auto ctx = assembler.Assemble(results, "query", 256);

    EXPECT_GT(ctx.sectionsIncluded, 0);
    EXPECT_GT(ctx.sectionsOmitted, 0);
    EXPECT_LT(ctx.sectionsIncluded, 50);
}

TEST_F(ContextAssemblerTest, EmptyResultsProduceEmptyContext) {
    std::vector<Core::SearchResult> results;
    auto ctx = assembler.Assemble(results, "query", 4096);

    EXPECT_EQ(ctx.sectionsIncluded, 0);
    EXPECT_TRUE(ctx.sourceMap.empty());
}

TEST_F(ContextAssemblerTest, SourceMapContainsCitations) {
    std::vector<Core::SearchResult> results = {
        MakeResult(7, "Risk Factors > Regional", "Asia-Pacific risks include..."),
    };

    auto ctx = assembler.Assemble(results, "risk", 4096);

    ASSERT_EQ(ctx.sourceMap.size(), 1);
    EXPECT_EQ(ctx.sourceMap[0].pageNumber, 7);
    EXPECT_EQ(ctx.sourceMap[0].sectionPath, "Risk Factors > Regional");
}

TEST_F(ContextAssemblerTest, CustomTokenEstimator) {
    // Custom estimator: 1 token per character (very aggressive)
    Application::ContextAssembler strictAssembler(
        [](const std::string& text) -> size_t { return text.size(); });

    std::vector<Core::SearchResult> results = {
        MakeResult(1, "Sec1", std::string(1000, 'a')),
        MakeResult(2, "Sec2", std::string(1000, 'b')),
    };

    // Budget of 500 tokens with 1-char-per-token should only fit header + partial results
    auto ctx = strictAssembler.Assemble(results, "query", 500);

    // At most the header + maybe one result due to tight budget
    EXPECT_LE(ctx.sectionsIncluded, 1);
}
