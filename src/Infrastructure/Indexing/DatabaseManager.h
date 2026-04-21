#pragma once
#include <sqlite3.h>
#include <string>
#include <filesystem>
#include <memory>
#include <functional>
#include <vector>
#include <optional>
#include <stdexcept>

namespace LocalNotebookLLM::Infrastructure {

    /// @brief RAII SQLite statement wrapper.
    class SqliteStatement {
    public:
        explicit SqliteStatement(sqlite3_stmt* stmt) : m_stmt(stmt) {}
        ~SqliteStatement() { if (m_stmt) sqlite3_finalize(m_stmt); }

        SqliteStatement(const SqliteStatement&) = delete;
        SqliteStatement& operator=(const SqliteStatement&) = delete;
        SqliteStatement(SqliteStatement&& o) noexcept : m_stmt(o.m_stmt) { o.m_stmt = nullptr; }
        SqliteStatement& operator=(SqliteStatement&& o) noexcept {
            if (this != &o) { if (m_stmt) sqlite3_finalize(m_stmt); m_stmt = o.m_stmt; o.m_stmt = nullptr; }
            return *this;
        }

        // ─── Bind parameters (1-indexed) ───
        void Bind(int idx, int64_t value)       { sqlite3_bind_int64(m_stmt, idx, value); }
        void Bind(int idx, int value)           { sqlite3_bind_int(m_stmt, idx, value); }
        void Bind(int idx, double value)        { sqlite3_bind_double(m_stmt, idx, value); }
        void Bind(int idx, const std::string& value) {
            sqlite3_bind_text(m_stmt, idx, value.c_str(), static_cast<int>(value.size()), SQLITE_TRANSIENT);
        }
        void BindNull(int idx) { sqlite3_bind_null(m_stmt, idx); }

        void BindBlob(int idx, const void* data, size_t size) {
            sqlite3_bind_blob(m_stmt, idx, data, static_cast<int>(size), SQLITE_TRANSIENT);
        }

        void Bind(int idx, std::optional<int64_t> value) {
            if (value) Bind(idx, *value); else BindNull(idx);
        }
        void Bind(int idx, std::optional<double> value) {
            if (value) Bind(idx, *value); else BindNull(idx);
        }

        // ─── Step / Execute ───
        bool Step() { return sqlite3_step(m_stmt) == SQLITE_ROW; }
        void Execute() {
            int rc = sqlite3_step(m_stmt);
            if (rc != SQLITE_DONE && rc != SQLITE_ROW) {
                throw std::runtime_error("SQLite execute failed: " +
                    std::string(sqlite3_errmsg(sqlite3_db_handle(m_stmt))));
            }
        }
        void Reset() { sqlite3_reset(m_stmt); sqlite3_clear_bindings(m_stmt); }

        // ─── Read columns (0-indexed) ───
        [[nodiscard]] int64_t     GetInt64(int col)  const { return sqlite3_column_int64(m_stmt, col); }
        [[nodiscard]] int         GetInt(int col)    const { return sqlite3_column_int(m_stmt, col); }
        [[nodiscard]] double      GetDouble(int col) const { return sqlite3_column_double(m_stmt, col); }
        [[nodiscard]] std::string GetString(int col) const {
            auto* text = reinterpret_cast<const char*>(sqlite3_column_text(m_stmt, col));
            return text ? text : "";
        }
        [[nodiscard]] bool IsNull(int col) const {
            return sqlite3_column_type(m_stmt, col) == SQLITE_NULL;
        }

        [[nodiscard]] sqlite3_stmt* GetHandle() const { return m_stmt; }

    private:
        sqlite3_stmt* m_stmt = nullptr;
    };

    /// @brief RAII transaction guard. Commits on Commit(), rolls back on destruction if not committed.
    class SqliteTransaction {
    public:
        explicit SqliteTransaction(sqlite3* db) : m_db(db) {
            sqlite3_exec(m_db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);
        }
        ~SqliteTransaction() { if (!m_committed) sqlite3_exec(m_db, "ROLLBACK", nullptr, nullptr, nullptr); }

        void Commit() {
            sqlite3_exec(m_db, "COMMIT", nullptr, nullptr, nullptr);
            m_committed = true;
        }

        SqliteTransaction(const SqliteTransaction&) = delete;
        SqliteTransaction& operator=(const SqliteTransaction&) = delete;

    private:
        sqlite3*  m_db = nullptr;
        bool      m_committed = false;
    };

    /// @brief Manages the SQLite database connection, schema initialization, and prepared statements.
    class DatabaseManager {
    public:
        /// @param dbPath Path to the SQLite database file. Use ":memory:" for in-memory databases.
        explicit DatabaseManager(const std::string& dbPath);
        ~DatabaseManager();

        DatabaseManager(const DatabaseManager&) = delete;
        DatabaseManager& operator=(const DatabaseManager&) = delete;
        DatabaseManager(DatabaseManager&&) noexcept;
        DatabaseManager& operator=(DatabaseManager&&) noexcept;

        /// Get the raw SQLite handle (for transactions etc.)
        [[nodiscard]] sqlite3* GetHandle() noexcept { return m_db; }

        /// Prepare a statement (caller owns the result).
        [[nodiscard]] SqliteStatement Prepare(const std::string& sql);

        /// Execute a statement that doesn't return rows.
        void Execute(const std::string& sql);

        /// Get the rowid of the last inserted row.
        [[nodiscard]] int64_t LastInsertRowId() const;

        /// Get the number of rows changed by the last statement.
        [[nodiscard]] int ChangeCount() const;

    private:
        sqlite3* m_db = nullptr;

        /// Load and execute Schema.sql to create/migrate tables.
        void InitializeSchema();

        /// Set performance-oriented PRAGMAs.
        void ConfigurePragmas();
    };

} // namespace LocalNotebookLLM::Infrastructure
