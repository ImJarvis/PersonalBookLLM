#include "Infrastructure/Indexing/DatabaseManager.h"
#include "Core/Log.h"
#include <stdexcept>
#include <fstream>
#include <sstream>

namespace LocalNotebookLLM::Infrastructure {

    // ─── Embedded schema (compiled-in from Schema.sql) ───
    static constexpr const char* SCHEMA_SQL = R"SQL(
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            filepath    TEXT NOT NULL UNIQUE,
            format      TEXT NOT NULL DEFAULT 'PDF',
            page_count  INTEGER NOT NULL DEFAULT 0,
            node_count  INTEGER NOT NULL DEFAULT 0,
            ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
            file_hash   TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);

        CREATE TABLE IF NOT EXISTS document_nodes (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            parent_id     INTEGER REFERENCES document_nodes(id),
            node_type     TEXT NOT NULL,
            page_number   INTEGER NOT NULL,
            section_path  TEXT NOT NULL,
            content       TEXT NOT NULL,
            font_size     REAL,
            is_bold       INTEGER DEFAULT 0,
            bbox_x        REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            confidence    REAL DEFAULT 1.0,
            sort_order    INTEGER NOT NULL,
            embedding     BLOB
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
            content,
            section_path,
            content='document_nodes',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2',
            prefix='2 3'
        );

        CREATE TRIGGER IF NOT EXISTS fts_insert AFTER INSERT ON document_nodes BEGIN
            INSERT INTO fts_index(rowid, content, section_path)
            VALUES (new.id, new.content, new.section_path);
        END;

        CREATE TRIGGER IF NOT EXISTS fts_delete AFTER DELETE ON document_nodes BEGIN
            INSERT INTO fts_index(fts_index, rowid, content, section_path)
            VALUES ('delete', old.id, old.content, old.section_path);
        END;

        CREATE TRIGGER IF NOT EXISTS fts_update AFTER UPDATE ON document_nodes BEGIN
            INSERT INTO fts_index(fts_index, rowid, content, section_path)
            VALUES ('delete', old.id, old.content, old.section_path);
            INSERT INTO fts_index(rowid, content, section_path)
            VALUES (new.id, new.content, new.section_path);
        END;

        CREATE INDEX IF NOT EXISTS idx_nodes_doc ON document_nodes(document_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_page ON document_nodes(document_id, page_number);
        CREATE INDEX IF NOT EXISTS idx_nodes_parent ON document_nodes(parent_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_sort ON document_nodes(document_id, sort_order);
    )SQL";

    DatabaseManager::DatabaseManager(const std::string& dbPath) {
        LOG_INFO("Database", "Opening database: " + dbPath);
        int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
        if (dbPath == ":memory:") {
            flags |= SQLITE_OPEN_MEMORY;
        }

        int rc = sqlite3_open_v2(dbPath.c_str(), &m_db, flags, nullptr);
        if (rc != SQLITE_OK) {
            std::string err = m_db ? sqlite3_errmsg(m_db) : "unknown error";
            if (m_db) sqlite3_close(m_db);
            m_db = nullptr;
            LOG_ERROR("Database", "Failed to open: " + err);
            throw std::runtime_error("Failed to open database '" + dbPath + "': " + err);
        }

        LOG_INFO("Database", "Configuring PRAGMAs...");
        ConfigurePragmas();
        LOG_INFO("Database", "Initializing schema...");
        InitializeSchema();
        LOG_INFO("Database", "Database ready");
    }

    DatabaseManager::~DatabaseManager() {
        if (m_db) {
            LOG_DEBUG("Database", "Closing database");
            sqlite3_close(m_db);
            m_db = nullptr;
        }
    }

    DatabaseManager::DatabaseManager(DatabaseManager&& o) noexcept
        : m_db(o.m_db) { o.m_db = nullptr; }

    DatabaseManager& DatabaseManager::operator=(DatabaseManager&& o) noexcept {
        if (this != &o) {
            if (m_db) sqlite3_close(m_db);
            m_db = o.m_db;
            o.m_db = nullptr;
        }
        return *this;
    }

    SqliteStatement DatabaseManager::Prepare(const std::string& sql) {
        sqlite3_stmt* stmt = nullptr;
        int rc = sqlite3_prepare_v2(m_db, sql.c_str(), static_cast<int>(sql.size()), &stmt, nullptr);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Failed to prepare SQL: " +
                std::string(sqlite3_errmsg(m_db)) + "\nSQL: " + sql);
        }
        return SqliteStatement(stmt);
    }

    void DatabaseManager::Execute(const std::string& sql) {
        char* errMsg = nullptr;
        int rc = sqlite3_exec(m_db, sql.c_str(), nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK) {
            std::string err = errMsg ? errMsg : "unknown error";
            sqlite3_free(errMsg);
            throw std::runtime_error("SQL execution failed: " + err);
        }
    }

    int64_t DatabaseManager::LastInsertRowId() const {
        return sqlite3_last_insert_rowid(m_db);
    }

    int DatabaseManager::ChangeCount() const {
        return sqlite3_changes(m_db);
    }

    void DatabaseManager::ConfigurePragmas() {
        // WAL mode for concurrent reads during writes
        Execute("PRAGMA journal_mode = WAL;");
        // Enable foreign keys
        Execute("PRAGMA foreign_keys = ON;");
        // Synchronous NORMAL is safe with WAL and much faster than FULL
        Execute("PRAGMA synchronous = NORMAL;");
        // 8 MB page cache (default is often 2 MB)
        Execute("PRAGMA cache_size = -8000;");
        // Memory-map 64 MB for faster reads
        Execute("PRAGMA mmap_size = 67108864;");
        // Temp tables in memory
        Execute("PRAGMA temp_store = MEMORY;");
    }

    void DatabaseManager::InitializeSchema() {
        Execute(SCHEMA_SQL);
    }

} // namespace LocalNotebookLLM::Infrastructure
