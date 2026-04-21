-- LocalNotebookLLM Database Schema
-- SQLite FTS5 with external content for zero-duplication structural indexing

-- Document metadata
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    filepath    TEXT NOT NULL UNIQUE,
    format      TEXT NOT NULL DEFAULT 'PDF',
    page_count  INTEGER NOT NULL DEFAULT 0,
    node_count  INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
    file_hash   TEXT NOT NULL  -- SHA-256 for dedup
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);

-- Structural nodes (the parsed tree, flattened for indexing)
CREATE TABLE IF NOT EXISTS document_nodes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id     INTEGER REFERENCES document_nodes(id),
    node_type     TEXT NOT NULL,      -- 'title','h1','h2','h3','paragraph','table','list_item'
    page_number   INTEGER NOT NULL,
    section_path  TEXT NOT NULL,       -- "Chapter 3 > 3.2 Risk Factors"
    content       TEXT NOT NULL,
    font_size     REAL,
    is_bold       INTEGER DEFAULT 0,
    bbox_x        REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
    confidence    REAL DEFAULT 1.0,
    sort_order    INTEGER NOT NULL     -- preserve document order
);

-- FTS5 index (external content — no data duplication)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
    content,
    section_path,
    content='document_nodes',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2',
    prefix='2 3'
);

-- Triggers to keep FTS5 in sync
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

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_nodes_doc ON document_nodes(document_id);
CREATE INDEX IF NOT EXISTS idx_nodes_page ON document_nodes(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON document_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_sort ON document_nodes(document_id, sort_order);
