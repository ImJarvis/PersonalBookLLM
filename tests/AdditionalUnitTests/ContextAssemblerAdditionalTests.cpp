#include <gtest/gtest.h>
#include "Application/ContextAssembler.h"
#include "Core/Enums/NodeType.h"

using namespace LocalNotebookLLM;

namespace {
Core::SearchResult MakeResult(int page, std::string section, std::string content, double score = 1.0) {
    Core::SearchResult r;
    r.nodeId = page * 10;
    r.documentId = 1;
    r.nodeType = Core::NodeType::Paragraph;
    r.pageNumber = page;
    r.sectionPath = std::move(section);
    r.content = std::move(content);
    r.bm25Score = score;
    return r;
}
}

TEST(ContextAssemblerAdditionalTest, FormatsContextWithPageTagAndSectionPath) {
    Application::ContextAssembler assembler;
    std::vector<Core::SearchResult> results{
        MakeResult(9, "Findings > Revenue", "Revenue increased 15% year-over-year.")
    };

    auto ctx = assembler.Assemble(results, "revenue", 2048);

    EXPECT_NE(ctx.formattedContext.find("[Page 9]"), std::string::npos);
    EXPECT_NE(ctx.formattedContext.find("Findings > Revenue"), std::string::npos);
    EXPECT_NE(ctx.formattedContext.find("Revenue increased 15%"), std::string::npos);
    ASSERT_EQ(ctx.sourceMap.size(), 1u);
    EXPECT_EQ(ctx.sourceMap[0].pageNumber, 9);
}

TEST(ContextAssemblerAdditionalTest, UsesDocumentRootWithoutSectionSuffix) {
    Application::ContextAssembler assembler;
    std::vector<Core::SearchResult> results{
        MakeResult(2, "Document Root", "Root level paragraph")
    };

    auto ctx = assembler.Assemble(results, "root", 2048);

    EXPECT_NE(ctx.formattedContext.find("[Page 2]"), std::string::npos);
    EXPECT_EQ(ctx.formattedContext.find("Document Root"), std::string::npos);
}

TEST(ContextAssemblerAdditionalTest, OmitsAllSectionsWhenTokenBudgetTooSmall) {
    Application::ContextAssembler strictAssembler(
        [](const std::string& text) -> size_t { return text.size(); });

    std::vector<Core::SearchResult> results{
        MakeResult(1, "SecA", std::string(400, 'a')),
        MakeResult(2, "SecB", std::string(400, 'b'))
    };

    auto ctx = strictAssembler.Assemble(results, "anything", 100);

    EXPECT_EQ(ctx.sectionsIncluded, 0);
    EXPECT_EQ(ctx.sectionsOmitted, 2);
    EXPECT_TRUE(ctx.sourceMap.empty());
}

TEST(ContextAssemblerAdditionalTest, TracksIncludedAndOmittedSections) {
    Application::ContextAssembler customAssembler(
        [](const std::string& text) -> size_t { return text.size() / 2; });

    std::vector<Core::SearchResult> results{
        MakeResult(1, "S1", std::string(200, 'x')),
        MakeResult(2, "S2", std::string(400, 'y')),
        MakeResult(3, "S3", std::string(600, 'z'))
    };

    auto ctx = customAssembler.Assemble(results, "query", 500);

    EXPECT_GT(ctx.sectionsIncluded, 0);
    EXPECT_GT(ctx.sectionsOmitted, 0);
}
