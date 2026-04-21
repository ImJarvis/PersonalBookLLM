#include <gtest/gtest.h>
#include "Infrastructure/Indexing/DatabaseManager.h"

using namespace LocalNotebookLLM::Infrastructure;

class DatabaseManagerTest : public ::testing::Test {
protected:
    std::unique_ptr<DatabaseManager> db;

    void SetUp() override {
        db = std::make_unique<DatabaseManager>(":memory:");
    }
};

TEST_F(DatabaseManagerTest, CreatesInMemoryDatabase) {
    EXPECT_NE(db->GetHandle(), nullptr);
}

TEST_F(DatabaseManagerTest, SchemaCreatesDocumentsTable) {
    auto stmt = db->Prepare("SELECT COUNT(*) FROM documents");
    EXPECT_TRUE(stmt.Step());
    EXPECT_EQ(stmt.GetInt(0), 0);
}

TEST_F(DatabaseManagerTest, SchemaCreatesDocumentNodesTable) {
    auto stmt = db->Prepare("SELECT COUNT(*) FROM document_nodes");
    EXPECT_TRUE(stmt.Step());
    EXPECT_EQ(stmt.GetInt(0), 0);
}

TEST_F(DatabaseManagerTest, SchemaCreatesFtsIndex) {
    // FTS5 virtual table should exist
    auto stmt = db->Prepare(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='fts_index'");
    EXPECT_TRUE(stmt.Step());
    EXPECT_EQ(stmt.GetInt(0), 1);
}

TEST_F(DatabaseManagerTest, InsertAndRetrieveDocument) {
    db->Execute(
        "INSERT INTO documents (filename, filepath, format, page_count, file_hash) "
        "VALUES ('test.pdf', '/path/test.pdf', 'PDF', 10, 'abc123')");

    auto id = db->LastInsertRowId();
    EXPECT_GT(id, 0);

    auto stmt = db->Prepare("SELECT filename, page_count FROM documents WHERE id = ?");
    stmt.Bind(1, id);
    EXPECT_TRUE(stmt.Step());
    EXPECT_EQ(stmt.GetString(0), "test.pdf");
    EXPECT_EQ(stmt.GetInt(1), 10);
}

TEST_F(DatabaseManagerTest, TransactionCommits) {
    {
        SqliteTransaction txn(db->GetHandle());
        db->Execute(
            "INSERT INTO documents (filename, filepath, format, page_count, file_hash) "
            "VALUES ('a.pdf', '/a.pdf', 'PDF', 5, 'hash1')");
        txn.Commit();
    }

    auto stmt = db->Prepare("SELECT COUNT(*) FROM documents");
    stmt.Step();
    EXPECT_EQ(stmt.GetInt(0), 1);
}

TEST_F(DatabaseManagerTest, TransactionRollsBackOnDestruction) {
    {
        SqliteTransaction txn(db->GetHandle());
        db->Execute(
            "INSERT INTO documents (filename, filepath, format, page_count, file_hash) "
            "VALUES ('b.pdf', '/b.pdf', 'PDF', 5, 'hash2')");
        // No Commit() — should rollback
    }

    auto stmt = db->Prepare("SELECT COUNT(*) FROM documents");
    stmt.Step();
    EXPECT_EQ(stmt.GetInt(0), 0);
}

TEST_F(DatabaseManagerTest, FtsTriggerInsertsOnNodeInsert) {
    // Insert a document first
    db->Execute(
        "INSERT INTO documents (filename, filepath, format, page_count, file_hash) "
        "VALUES ('test.pdf', '/test.pdf', 'PDF', 1, 'hash3')");
    auto docId = db->LastInsertRowId();

    // Insert a node
    auto stmt = db->Prepare(
        "INSERT INTO document_nodes "
        "(document_id, node_type, page_number, section_path, content, sort_order) "
        "VALUES (?, 'paragraph', 1, 'Intro', 'financial revenue quarterly results', 0)");
    stmt.Bind(1, docId);
    stmt.Execute();

    // FTS5 trigger should have auto-indexed the content
    auto search = db->Prepare("SELECT COUNT(*) FROM fts_index WHERE fts_index MATCH 'revenue'");
    search.Step();
    EXPECT_EQ(search.GetInt(0), 1);
}

TEST_F(DatabaseManagerTest, CascadeDeleteRemovesNodes) {
    db->Execute(
        "INSERT INTO documents (filename, filepath, format, page_count, file_hash) "
        "VALUES ('test.pdf', '/test.pdf', 'PDF', 1, 'hash4')");
    auto docId = db->LastInsertRowId();

    auto ins = db->Prepare(
        "INSERT INTO document_nodes "
        "(document_id, node_type, page_number, section_path, content, sort_order) "
        "VALUES (?, 'paragraph', 1, 'Test', 'content', 0)");
    ins.Bind(1, docId);
    ins.Execute();

    // Verify node exists
    auto check1 = db->Prepare("SELECT COUNT(*) FROM document_nodes");
    check1.Step();
    EXPECT_EQ(check1.GetInt(0), 1);

    // Delete document — should cascade
    auto del = db->Prepare("DELETE FROM documents WHERE id = ?");
    del.Bind(1, docId);
    del.Execute();

    auto check2 = db->Prepare("SELECT COUNT(*) FROM document_nodes");
    check2.Step();
    EXPECT_EQ(check2.GetInt(0), 0);
}
