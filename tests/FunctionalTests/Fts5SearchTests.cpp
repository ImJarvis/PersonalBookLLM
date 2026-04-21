#include <gtest/gtest.h>
#include "Infrastructure/Indexing/DatabaseManager.h"
#include "Infrastructure/Indexing/SqliteFts5Indexer.h"
#include "Infrastructure/Storage/SqliteDocumentRepository.h"
#include "Core/Models/DocumentTree.h"
#include "Core/Models/DocumentNode.h"
#include "Core/Enums/NodeType.h"

using namespace LocalNotebookLLM;

class Fts5SearchTest : public ::testing::Test {
protected:
    std::shared_ptr<Infrastructure::DatabaseManager>         db;
    std::shared_ptr<Infrastructure::SqliteFts5Indexer>       indexer;
    std::shared_ptr<Infrastructure::SqliteDocumentRepository> repo;

    void SetUp() override {
        db      = std::make_shared<Infrastructure::DatabaseManager>(":memory:");
        indexer = std::make_shared<Infrastructure::SqliteFts5Indexer>(db);
        repo    = std::make_shared<Infrastructure::SqliteDocumentRepository>(db);
    }

    /// Create a test document tree with known content.
    Core::DocumentTree MakeTestTree() {
        Core::DocumentNode root(Core::NodeType::Title, "Annual Report 2025", 1, 24.0f, true);

        Core::DocumentNode ch1(Core::NodeType::H1, "Financial Results", 3, 18.0f, true);
        ch1.AddChild(Core::DocumentNode(Core::NodeType::Paragraph,
            "Total revenue for Q3 was $4.2 million, representing a 15% increase.", 3));
        ch1.AddChild(Core::DocumentNode(Core::NodeType::Paragraph,
            "Operating margins improved to 23% from 19% in the prior year.", 4));

        Core::DocumentNode ch2(Core::NodeType::H1, "Risk Factors", 8, 18.0f, true);
        Core::DocumentNode sub21(Core::NodeType::H2, "Regional Analysis", 8, 14.0f, true);
        sub21.AddChild(Core::DocumentNode(Core::NodeType::Paragraph,
            "Asia-Pacific region faces currency volatility and regulatory uncertainty.", 9));
        sub21.AddChild(Core::DocumentNode(Core::NodeType::Paragraph,
            "European markets showed stable growth despite geopolitical tensions.", 10));
        ch2.AddChild(std::move(sub21));

        root.AddChild(std::move(ch1));
        root.AddChild(std::move(ch2));

        return Core::DocumentTree(std::move(root));
    }

    /// Ingest the test document and return its ID.
    int64_t IngestTestDocument() {
        Core::DocumentMetadata meta;
        meta.filename  = "annual_report.pdf";
        meta.filepath  = "/test/annual_report.pdf";
        meta.format    = Core::DocumentFormat::PDF;
        meta.pageCount = 10;
        meta.fileHash  = "test_hash_001";

        auto idResult = repo->Add(meta);
        EXPECT_TRUE(idResult.has_value());
        int64_t docId = idResult.value();

        auto tree = MakeTestTree();
        auto indexResult = indexer->IndexDocument(docId, tree);
        EXPECT_TRUE(indexResult.has_value());

        return docId;
    }
};

TEST_F(Fts5SearchTest, BasicKeywordSearch) {
    IngestTestDocument();

    auto results = indexer->Search("revenue");
    ASSERT_FALSE(results.empty());

    // The revenue paragraph should be found
    bool found = false;
    for (const auto& r : results) {
        if (r.content.find("revenue") != std::string::npos ||
            r.content.find("Revenue") != std::string::npos) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "Expected to find a result containing 'revenue'";
}

TEST_F(Fts5SearchTest, MultiKeywordSearch) {
    IngestTestDocument();

    auto results = indexer->Search("\"Asia\" OR \"currency\"");
    ASSERT_FALSE(results.empty());

    bool foundAsia = false;
    for (const auto& r : results) {
        if (r.content.find("Asia") != std::string::npos) {
            foundAsia = true;
            EXPECT_EQ(r.pageNumber, 9);
            break;
        }
    }
    EXPECT_TRUE(foundAsia);
}

TEST_F(Fts5SearchTest, ResultsHaveCorrectMetadata) {
    int64_t docId = IngestTestDocument();

    auto results = indexer->Search("revenue");
    ASSERT_FALSE(results.empty());

    auto& r = results[0];
    EXPECT_EQ(r.documentId, docId);
    EXPECT_GT(r.pageNumber, 0);
    EXPECT_FALSE(r.sectionPath.empty());
    EXPECT_FALSE(r.content.empty());
    EXPECT_GT(r.bm25Score, 0.0);
}

TEST_F(Fts5SearchTest, SearchRespectsTopK) {
    IngestTestDocument();

    Core::SearchOptions opts;
    opts.topK = 2;
    auto results = indexer->Search("\"revenue\" OR \"margin\" OR \"Asia\" OR \"European\"", opts);

    EXPECT_LE(results.size(), 2u);
}

TEST_F(Fts5SearchTest, SearchRespectsDocumentFilter) {
    int64_t docId = IngestTestDocument();

    Core::SearchOptions opts;
    opts.documentIdFilter = 999;  // Non-existent document
    auto results = indexer->Search("revenue", opts);

    EXPECT_TRUE(results.empty());

    opts.documentIdFilter = static_cast<int>(docId);
    results = indexer->Search("revenue", opts);
    EXPECT_FALSE(results.empty());
}

TEST_F(Fts5SearchTest, NoResultsForNonExistentTerm) {
    IngestTestDocument();
    auto results = indexer->Search("xyznonexistent");
    EXPECT_TRUE(results.empty());
}

TEST_F(Fts5SearchTest, IndexedNodeCountMatches) {
    IngestTestDocument();
    size_t count = indexer->GetIndexedNodeCount();
    EXPECT_GT(count, 0u);
}

TEST_F(Fts5SearchTest, RemoveDocumentCleansIndex) {
    int64_t docId = IngestTestDocument();

    // Verify we can find content
    auto before = indexer->Search("revenue");
    EXPECT_FALSE(before.empty());

    // Remove document
    indexer->RemoveDocument(docId);

    // Content should be gone
    auto after = indexer->Search("revenue");
    EXPECT_TRUE(after.empty());
    EXPECT_EQ(indexer->GetIndexedNodeCount(), 0u);
}

TEST_F(Fts5SearchTest, ReindexReplacesContent) {
    int64_t docId = IngestTestDocument();

    // Create a different tree
    Core::DocumentNode root(Core::NodeType::Document, "", 0);
    root.AddChild(Core::DocumentNode(Core::NodeType::Paragraph,
        "Completely different content about quantum physics.", 1));

    auto result = indexer->IndexDocument(docId, Core::DocumentTree(std::move(root)));
    EXPECT_TRUE(result.has_value());

    // Old content should be gone
    auto oldResults = indexer->Search("revenue");
    EXPECT_TRUE(oldResults.empty());

    // New content should be found
    auto newResults = indexer->Search("quantum");
    EXPECT_FALSE(newResults.empty());
}
