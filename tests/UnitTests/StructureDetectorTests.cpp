#include <gtest/gtest.h>
#include "Infrastructure/Parsing/StructureDetector.h"
#include "Core/Models/DocumentNode.h"
#include "Core/Enums/NodeType.h"

using namespace LocalNotebookLLM;
using namespace LocalNotebookLLM::Infrastructure;
using namespace LocalNotebookLLM::Core;

class StructureDetectorTest : public ::testing::Test {
protected:
    StructureDetector detector;

    /// Helper: create a raw text block as a parser would extract
    static RawTextBlock MakeBlock(const std::string& text, float fontSize,
                                  bool bold, float x, float y, int page = 1) {
        RawTextBlock b;
        b.text       = text;
        b.fontSize   = fontSize;
        b.isBold     = bold;
        b.bbox       = {x, y, 500.0f, fontSize + 4.0f};
        b.pageNumber = page;
        return b;
    }
};

// ─── NodeType Classification ───

TEST_F(StructureDetectorTest, LargeBoldTextClassifiedAsTitle) {
    auto block = MakeBlock("Annual Financial Report 2025", 24.0f, true, 72.0f, 72.0f);
    auto node = detector.Classify(block);
    EXPECT_EQ(node.GetType(), NodeType::Title);
    EXPECT_EQ(node.GetText(), "Annual Financial Report 2025");
    EXPECT_EQ(node.GetPageNumber(), 1);
}

TEST_F(StructureDetectorTest, MediumBoldTextClassifiedAsH1) {
    auto block = MakeBlock("1. Introduction", 18.0f, true, 72.0f, 150.0f);
    auto node = detector.Classify(block);
    EXPECT_EQ(node.GetType(), NodeType::H1);
}

TEST_F(StructureDetectorTest, SmallBoldTextClassifiedAsH2) {
    auto block = MakeBlock("1.1 Background", 14.0f, true, 72.0f, 200.0f);
    auto node = detector.Classify(block);
    EXPECT_EQ(node.GetType(), NodeType::H2);
}

TEST_F(StructureDetectorTest, NormalTextClassifiedAsParagraph) {
    auto block = MakeBlock(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        11.0f, false, 72.0f, 250.0f);
    auto node = detector.Classify(block);
    EXPECT_EQ(node.GetType(), NodeType::Paragraph);
}

TEST_F(StructureDetectorTest, BulletPrefixClassifiedAsListItem) {
    auto block = MakeBlock("- Revenue increased by 15%", 11.0f, false, 90.0f, 300.0f);
    auto node = detector.Classify(block);
    EXPECT_EQ(node.GetType(), NodeType::ListItem);
}

TEST_F(StructureDetectorTest, NumberedPrefixClassifiedAsListItem) {
    auto block = MakeBlock("3. Update the configuration file", 11.0f, false, 90.0f, 300.0f);
    auto node = detector.Classify(block);
    EXPECT_EQ(node.GetType(), NodeType::ListItem);
}

// ─── Tree Construction ───

TEST_F(StructureDetectorTest, BuildTreeMaintainsHierarchy) {
    std::vector<RawTextBlock> blocks = {
        MakeBlock("Report Title", 24.0f, true, 72.0f, 72.0f),
        MakeBlock("Chapter 1", 18.0f, true, 72.0f, 150.0f),
        MakeBlock("Some paragraph text under chapter 1.", 11.0f, false, 72.0f, 200.0f),
        MakeBlock("1.1 Subsection", 14.0f, true, 72.0f, 260.0f),
        MakeBlock("Content under subsection 1.1.", 11.0f, false, 72.0f, 310.0f),
        MakeBlock("Chapter 2", 18.0f, true, 72.0f, 400.0f),
        MakeBlock("Content under chapter 2.", 11.0f, false, 72.0f, 450.0f),
    };

    auto tree = detector.BuildTree(blocks);
    auto& root = tree.GetRoot();

    // Root should be the Title
    EXPECT_EQ(root.GetType(), NodeType::Title);
    EXPECT_EQ(root.GetText(), "Report Title");

    // Title has 2 children: Chapter 1 and Chapter 2
    ASSERT_EQ(root.GetChildren().size(), 2);

    auto& ch1 = root.GetChildren()[0];
    EXPECT_EQ(ch1.GetType(), NodeType::H1);
    EXPECT_EQ(ch1.GetText(), "Chapter 1");

    // Chapter 1 has 2 children: a paragraph and subsection 1.1
    ASSERT_EQ(ch1.GetChildren().size(), 2);
    EXPECT_EQ(ch1.GetChildren()[0].GetType(), NodeType::Paragraph);

    auto& sub11 = ch1.GetChildren()[1];
    EXPECT_EQ(sub11.GetType(), NodeType::H2);
    EXPECT_EQ(sub11.GetText(), "1.1 Subsection");

    // Subsection has 1 child paragraph
    ASSERT_EQ(sub11.GetChildren().size(), 1);
    EXPECT_EQ(sub11.GetChildren()[0].GetType(), NodeType::Paragraph);

    // Chapter 2
    auto& ch2 = root.GetChildren()[1];
    EXPECT_EQ(ch2.GetType(), NodeType::H1);
    ASSERT_EQ(ch2.GetChildren().size(), 1);
}

// ─── Edge Cases ───

TEST_F(StructureDetectorTest, EmptyBlockListProducesEmptyTree) {
    auto tree = detector.BuildTree({});
    EXPECT_TRUE(tree.IsEmpty());
}

TEST_F(StructureDetectorTest, SingleParagraphNoHeadingCreatesDocRoot) {
    std::vector<RawTextBlock> blocks = {
        MakeBlock("Just a paragraph with no heading.", 11.0f, false, 72.0f, 72.0f),
    };

    auto tree = detector.BuildTree(blocks);
    auto& root = tree.GetRoot();

    // With no title/heading, root is a synthetic "Document" node
    EXPECT_EQ(root.GetType(), NodeType::Document);
    ASSERT_EQ(root.GetChildren().size(), 1);
    EXPECT_EQ(root.GetChildren()[0].GetType(), NodeType::Paragraph);
}

TEST_F(StructureDetectorTest, GarbageTextDetectedLowConfidence) {
    auto block = MakeBlock("\xC3\x98\xC2\xA3\xC2\xA4\xC2\xA7\xC2\xBD\xC2\xBE\xC2\xBF", 
                           11.0f, false, 72.0f, 72.0f);
    auto node = detector.Classify(block);
    EXPECT_LE(node.GetConfidence(), 0.5f);
}

TEST_F(StructureDetectorTest, PageNumberPreserved) {
    std::vector<RawTextBlock> blocks = {
        MakeBlock("Page 1 Content", 11.0f, false, 72.0f, 72.0f, 1),
        MakeBlock("Page 2 Content", 11.0f, false, 72.0f, 72.0f, 2),
    };

    auto tree = detector.BuildTree(blocks);
    auto allNodes = tree.Flatten();

    bool hasPage1 = false, hasPage2 = false;
    for (const auto& n : allNodes) {
        if (n.GetPageNumber() == 1) hasPage1 = true;
        if (n.GetPageNumber() == 2) hasPage2 = true;
    }
    EXPECT_TRUE(hasPage1);
    EXPECT_TRUE(hasPage2);
}

TEST_F(StructureDetectorTest, TreeNodeCount) {
    std::vector<RawTextBlock> blocks = {
        MakeBlock("Title", 24.0f, true, 72.0f, 72.0f),
        MakeBlock("Heading", 18.0f, true, 72.0f, 150.0f),
        MakeBlock("Para 1", 11.0f, false, 72.0f, 200.0f),
        MakeBlock("Para 2", 11.0f, false, 72.0f, 250.0f),
    };

    auto tree = detector.BuildTree(blocks);
    EXPECT_EQ(tree.NodeCount(), 4);  // Title + H1 + 2 Paragraphs
}
