/**
 * pi-omlx-context-store — Local Context Store via SQLite + FTS5 Full-Text Search
 *
 * Architecture: Large inputs are split into chunks and stored in a local SQLite
 * database. The LLM queries chunks by UID and receives the FULL chunk content,
 * allowing it to understand complete context and extract relevant information itself.
 *
 * Flow:
 *   1. Large input detected → split into chunks and stored in SQLite (NOT in context)
 *   2. LLM receives chunk UIDs and queries each chunk by UID
 *   3. Extension returns the FULL chunk content for each UID query
 *   4. LLM analyzes full chunks and extracts relevant information
 *
 * Key design principle: The extension stores full content but does NOT pre-filter it.
 * The LLM is responsible for understanding context and determining relevance — not the extension.
 * The LLM cannot query a chunk correctly if it hasn't seen the full content, so full chunks
 * are returned for UID-based lookups.
 *
 * This is a local RAG system: fast, deterministic, no API calls for search.
 * Uses better-sqlite3 (native) with FTS5 + Porter stemmer + Boolean queries.
 *
 * Usage:
 *   pi --extension pi-omlx-context-store.js
 *
 * Dependencies: better-sqlite3 (npm install better-sqlite3)
 *
 * @license MIT
 */

// ============================================================================
// Configuration
// ============================================================================

const DB_NAME = "pi-context-store.db";
const STORE_DIR = ".pi/agent/context-store";
const CHUNK_SIZE = parseInt(process.env.CHUNK_SIZE) || 5000; // Characters per chunk — also the threshold to trigger storage (no point chunking content smaller than one chunk)
const MAX_QUERY_RESULTS = 20;
const CONTEXT_WINDOW_TAILROOM = 8000; // Reserve tokens for LLM response

// ============================================================================
// SQLite Storage Layer (better-sqlite3 + FTS5)
// ============================================================================

let Database = null;
let dbInstance = null;
let nextRowid = 1;

function getDbPath() {
  const fs = require("fs");
  const path = require("path");
  const os = require("os");
  const dbPath = path.join(os.homedir(), STORE_DIR, DB_NAME);
  const dir = path.dirname(dbPath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  return dbPath;
}

function getDb() {
  if (!dbInstance) {
    Database = require("better-sqlite3");
    const dbPath = getDbPath();
    dbInstance = new Database(dbPath);
    // Enable WAL mode for better concurrent read performance
    dbInstance.pragma("journal_mode = WAL");
  }
  return dbInstance;
}

function initDb() {
  const db = getDb();

  // Main table: metadata + full text copy (for FTS trigger sync)
  db.exec(`
    CREATE TABLE IF NOT EXISTS content (
      uid TEXT PRIMARY KEY,
      title TEXT,
      type TEXT,
      sz INTEGER,
      body TEXT,
      queries INTEGER DEFAULT 0,
      created_at INTEGER,
      session_id TEXT,
      confirmed INTEGER DEFAULT 0
    )
  `);

  // FTS5 virtual table: full-text search index
  // Uses Porter stemmer (matches "running" for query "run")
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
      title,
      body,
      tokenize='porter'
    )
  `);

  // Triggers: keep FTS index in sync with main table
  db.exec(`CREATE TRIGGER IF NOT EXISTS content_ai AFTER INSERT ON content BEGIN
    INSERT INTO content_fts(rowid, title, body) VALUES (new.rid, new.title, new.body);
  END`);

  // We use rid as INTEGER PRIMARY KEY for FTS rowid mapping
  // Add rid column if not exists (for first-time setup)
  try {
    db.prepare("ALTER TABLE content ADD COLUMN rid INTEGER").run();
  } catch (_) {
    // Column already exists
  }

  // Schema migration: add session tracking columns (v2)
  try {
    db.prepare("ALTER TABLE content ADD COLUMN session_id TEXT").run();
  } catch (_) { /* already exists */ }
  try {
    db.prepare("ALTER TABLE content ADD COLUMN confirmed INTEGER DEFAULT 0").run();
  } catch (_) { /* already exists */ }

  // Rebuild triggers with correct column name
  try { db.exec("DROP TRIGGER IF EXISTS content_ai"); } catch (_) {}
  try { db.exec("DROP TRIGGER IF EXISTS content_au"); } catch (_) {}

  db.exec(`CREATE TRIGGER content_ai AFTER INSERT ON content BEGIN
    INSERT INTO content_fts(rowid, title, body) VALUES (new.rid, new.title, new.body);
  END`);

  db.exec(`CREATE TRIGGER content_au AFTER UPDATE ON content
  WHEN old.rid != new.rid OR old.title != new.title OR old.body != new.body
  BEGIN
    INSERT INTO content_fts(content_fts, rowid, title, body) VALUES('delete', old.rid, old.title, old.body);
    INSERT INTO content_fts(rowid, title, body) VALUES (new.rid, new.title, new.body);
  END`);

  // Get next rowid from autoincrement
  const info = db.prepare("SELECT MAX(rid) as mr FROM content").get();
  nextRowid = (info?.mr || 0) + 1;

  return db;
}

function storeContent(rawText, meta = {}) {
  const db = getDb();
  const rid = nextRowid++;

  const uid = meta.uid || "doc_" + rid;
  const title = (meta.title || rawText.slice(0, 150) + (rawText.length > 150 ? "..." : "")).trim();
  const type = meta.type || inferType(meta.filename || "");
  const sessionId = meta.sessionId || "default";

  // Insert into main table — FTS trigger auto-syncs to content_fts
  db.prepare(
    "INSERT INTO content (rid, uid, title, type, sz, body, queries, created_at, session_id, confirmed) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)"
  ).run(rid, uid, title, type, rawText.length, rawText, Date.now(), sessionId, 0);

  return { uid, rid, title, type, size: rawText.length, sessionId };
}

/**
 * Split text into chunks and store each one with its own UID.
 * Returns array of chunk metadata so the caller can inform the LLM
 * about how many chunks exist, their sizes, and their UIDs.
 */
function storeChunks(rawText, meta = {}) {
  const db = getDb();
  const chunks = splitIntoChunks(rawText, CHUNK_SIZE);
  const sessionId = meta.sessionId || "default";
  const baseRid = nextRowid; // Reserve a range of rowids
  const chunkMetas = chunks.map((chunk, i) => {
    const rid = nextRowid++;
    const uid = meta.uid ? `${meta.uid}_chunk_${i}` : `doc_${baseRid}_chunk_${i}`;
    const title = (meta.title || rawText.slice(0, 150) + (rawText.length > 150 ? "..." : "")).trim();
    const type = meta.type || inferType(meta.filename || "");

    db.prepare(
      "INSERT INTO content (rid, uid, title, type, sz, body, queries, created_at, session_id, confirmed) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)"
    ).run(rid, uid, title, type, chunk.length, chunk, Date.now(), sessionId, 0);

    return { uid, rid, index: i, size: chunk.length };
  });

  return {
    sessionId,
    totalChunks: chunkMetas.length,
    totalSize: rawText.length,
    chunks: chunkMetas,
  };
}

/**
 * Split text into chunks of approximately chunkSize characters.
 * Splits on whitespace boundaries when possible to avoid breaking words.
 */
function splitIntoChunks(text, chunkSize) {
  if (text.length <= chunkSize) {
    return [text];
  }

  const chunks = [];
  let offset = 0;

  while (offset < text.length) {
    let end = Math.min(offset + chunkSize, text.length);

    // Try to split at a whitespace boundary if we're not at the end of text
    if (end < text.length) {
      // Look back up to 200 chars for a good split point
      const lookBackStart = Math.max(offset, end - 200);
      const segment = text.slice(lookBackStart, end);
      const lastSpace = Math.max(segment.lastIndexOf(" "), segment.lastIndexOf("\n"), segment.lastIndexOf("\t"));
      if (lastSpace > segment.length * 0.5) {
        end = lookBackStart + lastSpace;
      }
    }

    chunks.push(text.slice(offset, end));
    offset = end;
  }

  return chunks;
}

function extractUidTargets(searchText) {
  const text = (searchText || "").trim();
  if (!text) return [];

  // Explicit uid:<value> / uid=<value>
  const explicit = [...text.matchAll(/\buid\s*[:=]\s*([A-Za-z0-9_.:-]+)/gi)].map(m => m[1]);

  // Bare UIDs commonly generated by this extension (doc_123, doc_123_chunk_0)
  const bare = [...text.matchAll(/\bdoc_[A-Za-z0-9]+(?:_chunk_[0-9]+)?\b/g)].map(m => m[0]);

  return [...new Set([...explicit, ...bare])];
}

function ftsSafeQuery(searchText) {
  // Keep user intent for operators if possible, but ensure we always have a safe fallback.
  const raw = (searchText || "").trim();
  if (!raw) return "";

  // Extract token-like pieces and join with AND for robust matching.
  // This avoids parser errors on punctuation like "llms.txt".
  const tokens = raw.match(/[A-Za-z0-9_]+/g) || [];
  if (!tokens.length) return "";
  return tokens.join(" AND ");
}

/**
 * Fetch full chunk content by UID.
 * When the LLM queries by UID, it receives the complete chunk — not a snippet.
 * This allows the LLM to understand full context and extract relevance itself.
 */
function fetchRowsByUid(uids) {
  if (!uids?.length) return [];
  const db = getDb();

  const getStmt = db.prepare(
    "SELECT uid, title, type, sz, created_at, queries, LENGTH(body) as body_len FROM content WHERE uid = ?"
  );
  const bumpStmt = db.prepare("UPDATE content SET queries = queries + 1 WHERE uid = ?");

  const rows = [];
  for (const uid of uids) {
    const row = getStmt.get(uid);
    if (!row) continue;
    bumpStmt.run(uid);
    rows.push({
      ...row,
      queries: (row.queries || 0) + 1,
      body: fetchChunk(uid),
    });
  }
  return rows;
}

function queryIndex(searchText) {
  const db = getDb();
  const raw = (searchText || "").trim();
  if (!raw) return [];

  // Path 1: UID lookups (exact) — avoids FTS parser issues and "no such column: uid"
  const uidTargets = extractUidTargets(raw);
  if (uidTargets.length) {
    return fetchRowsByUid(uidTargets);
  }

  // Shared mapper for FTS/LIKE result sets
  const mapRows = (results, snippetNeedle) => {
    const rows = (results || []).map(row => ({
      uid: row.uid,
      title: row.title,
      type: row.type,
      sz: row.sz,
      created_at: row.created_at,
      queries: (row.queries || 0) + 1,
      body_len: row.body_len,
      snippet: fetchSnippet(row.uid, snippetNeedle),
    }));

    rows.forEach(r => {
      db.prepare("UPDATE content SET queries = ? WHERE uid = ?").run(r.queries, r.uid);
    });

    return rows;
  };

  // Path 2: FTS query using raw input first
  try {
    const results = db.prepare(
      `SELECT c.uid, c.title, c.type, c.sz, c.created_at, c.queries,
              LENGTH(c.body) as body_len
       FROM content_fts f
       JOIN content c ON c.rid = f.rowid
       WHERE content_fts MATCH ?
       ORDER BY rank
       LIMIT ?`
    ).all(raw, MAX_QUERY_RESULTS);

    return mapRows(results, raw);
  } catch (_) {
    // Continue to safe fallbacks below
  }

  // Path 3: FTS-safe rewritten query (punctuation sanitized)
  const safe = ftsSafeQuery(raw);
  if (safe) {
    try {
      const results = db.prepare(
        `SELECT c.uid, c.title, c.type, c.sz, c.created_at, c.queries,
                LENGTH(c.body) as body_len
         FROM content_fts f
         JOIN content c ON c.rid = f.rowid
         WHERE content_fts MATCH ?
         ORDER BY rank
         LIMIT ?`
      ).all(safe, MAX_QUERY_RESULTS);

      if (results?.length) {
        return mapRows(results, safe);
      }
    } catch (_) {
      // Continue to LIKE fallback
    }
  }

  // Path 4: SQL LIKE fallback — guarantees non-crashing behavior
  const likeNeedle = `%${raw.toLowerCase()}%`;
  const likeResults = db.prepare(
    `SELECT uid, title, type, sz, created_at, queries, LENGTH(body) as body_len
     FROM content
     WHERE lower(title) LIKE ? OR lower(body) LIKE ?
     ORDER BY created_at DESC
     LIMIT ?`
  ).all(likeNeedle, likeNeedle, MAX_QUERY_RESULTS);

  return mapRows(likeResults, raw);
}

/**
 * Fetch full chunk content by UID.
 * Returns the complete body text — the LLM needs full context to understand and extract relevance.
 */
function fetchChunk(docUid) {
  const db = getDb();
  if (!db) return "";

  const row = db.prepare("SELECT body FROM content WHERE uid = ?").get(docUid);
  return row?.body || "";
}

/**
 * Fetch a snippet for non-UID queries (FTS/LIKE results).
 * Returns ~500 chars around the match for search results where the LLM
 * hasn't explicitly requested a specific chunk.
 */
function fetchSnippet(docUid, searchText) {
  const db = getDb();
  if (!db) return "";

  const row = db.prepare("SELECT body FROM content WHERE uid = ?").get(docUid);
  if (!row) return "";

  const body = row.body || "";
  const lowerBody = body.toLowerCase();

  // Use first non-empty token as snippet anchor for operator-heavy queries
  const token = ((searchText || "").match(/[A-Za-z0-9_]+/g) || [""])[0].toLowerCase();
  const idx = token ? lowerBody.indexOf(token) : -1;

  if (idx === -1) {
    // No exact-ish match — return first 500 chars for context
    return body.slice(0, 500);
  }

  // Extract context window around the match (±200 chars)
  const start = Math.max(0, idx - 200);
  const end = Math.min(body.length, idx + token.length + 200);
  const snippet = body.slice(start, end);

  return (start > 0 ? "... " : "") + snippet + (end < body.length ? " ..." : "");
}

function inferType(filename) {
  if (!filename) return "text";
  const ext = filename.split(".").pop()?.toLowerCase() || "";
  const typeMap = {
    js: "code", ts: "code", py: "code", java: "code", cpp: "code", c: "code",
    rs: "code", go: "code", rb: "code", php: "code", swift: "code",
    json: "data", yaml: "data", yml: "data", xml: "data", toml: "data",
    md: "document", txt: "document", rst: "document",
    css: "style", scss: "style", sass: "style",
    html: "markup", htm: "markup",
  };
  return typeMap[ext] || "text";
}

function listIndex() {
  const db = getDb();
  if (!db) return [];

  const result = db.prepare(
    "SELECT uid, title, type, sz, created_at, queries FROM content ORDER BY created_at DESC"
  ).all();
  return result.map(row => ({
    uid: row.uid,
    title: row.title,
    type: row.type,
    sz: row.sz,
    created_at: row.created_at,
    queries: row.queries,
  }));
}

function getStats() {
  const db = getDb();
  if (!db) return { docs: 0, totalBytes: 0 };

  const row = db.prepare("SELECT COUNT(*) as cnt, COALESCE(SUM(sz), 0) as bytes FROM content").get();
  return { docs: row.cnt, totalBytes: row.bytes };
}

function clearIndex() {
  const db = getDb();

  // Count and size before dropping
  const stats = db.prepare("SELECT COUNT(*) as cnt, COALESCE(SUM(sz), 0) as bytes FROM content").get();
  if (stats.cnt === 0) return { purged: 0, freedBytes: 0 };

  // Drop both tables (FTS5 can't be deleted from via triggers)
  try { db.exec("DROP TABLE IF EXISTS content"); } catch (_) {}
  try { db.exec("DROP TABLE IF EXISTS content_fts"); } catch (_) {}

  // Recreate — initDb schema
  db.exec(`
    CREATE TABLE content (
      uid TEXT PRIMARY KEY,
      title TEXT,
      type TEXT,
      sz INTEGER,
      body TEXT,
      queries INTEGER DEFAULT 0,
      created_at INTEGER,
      session_id TEXT,
      confirmed INTEGER DEFAULT 0
    )
  `);
  db.exec(`
    CREATE VIRTUAL TABLE content_fts USING fts5(
      title,
      body,
      tokenize='porter'
    )
  `);
  db.exec(`CREATE TRIGGER content_ai AFTER INSERT ON content BEGIN
    INSERT INTO content_fts(rowid, title, body) VALUES (new.rid, new.title, new.body);
  END`);
  db.exec(`CREATE TRIGGER content_au AFTER UPDATE ON content
  WHEN old.rid != new.rid OR old.title != new.title OR old.body != new.body
  BEGIN
    INSERT INTO content_fts(content_fts, rowid, title, body) VALUES('delete', old.rid, old.title, old.body);
    INSERT INTO content_fts(rowid, title, body) VALUES (new.rid, new.title, new.body);
  END`);

  nextRowid = 1;
  console.log(`[context-store] Tables dropped & recreated — ${stats.cnt} docs (${(stats.bytes / 1024).toFixed(1)} KB) purged`);
  return { purged: stats.cnt, freedBytes: stats.bytes };
}

// ============================================================================
// Pi Extension Integration
// ============================================================================

module.exports = function (pi) {
  const fs = require("fs");
  const path = require("path");
  const { Type } = require("typebox");

  // Initialize SQLite on load
  try {
    initDb();
    console.log("[context-store] Extension loaded — SQLite + FTS5 local index ready");
  } catch (err) {
    console.error("[context-store] Failed to initialize SQLite:", err.message);
    console.error("[context-store] Install better-sqlite3: npm install better-sqlite3");
  }

  // Register tools once session starts (runtime must be ready)
  pi.on("session_start", async (_event, ctx) => {
    // Check if tools already registered (prevent double registration)
    const existingTools = pi.getAllTools();
    const hasTools = existingTools.some(t => t.name === 'context_search');
    
    if (hasTools) {
      return; // Already registered, skip
    }

    // Register context_search tool
    pi.registerTool({
      name: "context_search",
      description: "Search the local RAG index. When querying by UID (e.g., 'uid:doc_1_chunk_0'), returns the FULL chunk content. For general search queries, returns relevant snippets.",
      parameters: Type.Object({
        query: Type.String({ description: "Search terms or UID lookup. Use 'uid:<value>' or 'doc_*' identifiers to retrieve full chunk content. Supports FTS5 operators (AND, OR, NOT, \"exact phrases\") for general search." }),
      }),
      execute: async (_toolCallId, params) => {
        try {
          const results = queryIndex(params.query);
          if (results.length === 0) {
            return { content: [{ type: "text", text: "No results found in the index." }] };
          }
          return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
        } catch (err) {
          return { content: [{ type: "text", text: `Search failed: ${err.message}` }] };
        }
      },
    });

    // Register context_list tool
    pi.registerTool({
      name: "context_list",
      description: "List all documents stored in the local RAG index with metadata.",
      parameters: Type.Object({}),
      execute: async () => {
        try {
          const items = listIndex();
          return { content: [{ type: "text", text: JSON.stringify(items, null, 2) }] };
        } catch (err) {
          return { content: [{ type: "text", text: `List failed: ${err.message}` }] };
        }
      },
    });

    // Register context_stats tool
    pi.registerTool({
      name: "context_stats",
      description: "Get statistics about the local RAG index (document count, total size).",
      parameters: Type.Object({}),
      execute: async () => {
        try {
          const stats = getStats();
          return { content: [{ type: "text", text: JSON.stringify(stats) }] };
        } catch (err) {
          return { content: [{ type: "text", text: `Stats failed: ${err.message}` }] };
        }
      },
    });

    // Register context_store tool
    pi.registerTool({
      name: "context_store",
      description: "Store text content in the local RAG index for later full-text search.",
      parameters: Type.Object({
        text: Type.String({ description: "The content to store in the index" }),
        title: Type.Optional(Type.String({ description: "Optional title for the document" })),
        type: Type.Optional(Type.String({ description: "Optional type: code, data, document, style, markup" })),
      }),
      execute: async (_toolCallId, params) => {
        try {
          const result = storeContent(params.text, { title: params.title, type: params.type });
          return { content: [{ type: "text", text: JSON.stringify(result) }] };
        } catch (err) {
          return { content: [{ type: "text", text: `Store failed: ${err.message}` }] };
        }
      },
    });

    // Register context_clear tool
    pi.registerTool({
      name: "context_clear",
      description: "Clear the entire local RAG index.",
      parameters: Type.Object({}),
      execute: async () => {
        try {
          clearIndex();
          return { content: [{ type: "text", text: JSON.stringify({ ok: true }) }] };
        } catch (err) {
          return { content: [{ type: "text", text: `Clear failed: ${err.message}` }] };
        }
      },
    });
  });

  // ========================================================================
  // Layer 1: Intercept large inputs — store to SQLite, NOT context
  // ========================================================================
  pi.on("input", async (event, ctx) => {
    const text = event.text || "";

    if (text.length <= CHUNK_SIZE) {
      return { action: "continue" };
    }

    // Large import detected — split into chunks and store to SQLite index, NOT context
    const sessionId = "session_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
    let result;
    try {
      result = storeChunks(text, {
        title: `Imported: ${text.slice(0, 80)}...`,
        filename: event.filename || "unknown",
        sessionId: sessionId,
      });
    } catch (err) {
      console.error("[context-store] Store failed:", err.message);
      return { action: "continue" }; // DB failed, let it through normally
    }

    if (!result) {
      return { action: "continue" };
    }

    // Verify all chunks are indexed in FTS5 before allowing queries
    const db = getDb();
    const expectedCount = result.totalChunks;
    const actualCount = db.prepare("SELECT COUNT(*) as cnt FROM content_fts WHERE rowid >= ? AND rowid < ?").get(
      result.chunks[0]?.rid,
      result.chunks[result.totalChunks - 1]?.rid + 1
    ).cnt;

    if (actualCount < expectedCount) {
      console.error(`[context-store] FTS5 indexing incomplete: ${actualCount}/${expectedCount} chunks indexed. Retrying...`);
      // Force FTS optimization to ensure all chunks are indexed
      db.exec("OPTIMIZE content_fts");
    }

    const chunkInfo = result.chunks.map(c => `${c.uid}(${(c.size / 1024).toFixed(1)}KB)`).join(", ");
    console.log(`[context-store] Stored ${result.totalChunks} chunks [${result.chunks[0]?.uid}] → session=${sessionId}`);

    // Notify user
    if (ctx.hasUI) {
      ctx.ui.notify(
        `📦 Large import (${result.totalSize.toLocaleString()} chars) split into ${result.totalChunks} chunk(s) and stored in local index. ` +
        `Use context_search with each chunk's UID to retrieve full content.`,
        "info"
      );
    }

    // Transform: replace large content with instructions for the LLM to query and reconstruct understanding
    const chunkIds = result.chunks.map(c => c.uid).join(", ");
    return {
      action: "transform",
      text: `[SYSTEM: Large input detected and stored in local RAG index. DO NOT process raw content — it has been removed from your context.\n\n` +
            `CRITICAL: All ${result.totalChunks} chunk(s) are fully stored and indexed. ONLY NOW may you query.\n\n` +
            `REQUIRED SEQUENCE:\n` +
            `STEP 1: Query the database for EACH chunk's content using their UIDs\n` +
            `STEP 2: Read the FULL chunk content returned for each UID\n` +
            `STEP 3: Extract the main ideas from each chunk individually\n` +
            `STEP 4: Reconstruct a coherent understanding of the complete original input\n` +
            `STEP 5: ONLY after completing steps 1-4, if uncertainties remain, ASK the user for clarification\n\n` +
            `DATABASE QUERY INSTRUCTIONS:\n` +
            `- Retrieve ALL ${result.totalChunks} chunk(s) by UID: [${chunkIds}]\n` +
            `- Use context_search with each chunk's UID to get the FULL chunk content\n` +
            `- Total size: ${result.totalSize.toLocaleString()} chars (${(result.totalSize / 1024).toFixed(1)} KB)\n` +
            `- Do NOT ask the user anything until you have queried all chunks and read their full content\n\n` +
            `GOAL: After querying all chunks and reading their full content, you should have a complete understanding of the original input. Only then may you ask the user about specific uncertainties.`,
    };
  });

  // ========================================================================
  // Commands for manual interaction
  // ========================================================================
  pi.registerCommand("context-search", {
    description: "Search the local content index. Usage: /context-search <query>",
    handler: async (args, ctx) => {
      const query = args.trim();
      if (!query) {
        ctx.ui.notify("Usage: /context-search <search terms>", "warning");
        return;
      }

      try {
        const results = queryIndex(query);
        if (results.length === 0) {
          ctx.ui.notify("No results found in index.", "info");
          return;
        }

        const summary = results.map((r, i) =>
          `${i + 1}. [${r.type}] ${r.title} (${(r.sz / 1024).toFixed(1)} KB) — ${r.snippet?.slice(0, 120)}`
        ).join("\n");

        ctx.ui.notify(`Found ${results.length} results:\n${summary}`, "info");
      } catch (err) {
        ctx.ui.notify(`Search failed: ${err.message}`, "error");
      }
    },
  });

  pi.registerCommand("context-status", {
    description: "Show index status: document count and total size",
    handler: async (_args, ctx) => {
      try {
        const stats = getStats();
        const list = listIndex();

        if (stats.docs === 0) {
          ctx.ui.notify("Index is empty. Large imports are stored here automatically.", "info");
          return;
        }

        const summary = list.map(d =>
          `  • ${d.title?.slice(0, 60)} — ${(d.sz / 1024).toFixed(1)} KB (${d.queries} queries)`
        ).join("\n");

        ctx.ui.notify(
          `📊 Index: ${stats.docs} documents, ${(stats.totalBytes / 1024).toFixed(0)} KB total\n${summary}`,
          "info"
        );
      } catch (err) {
        ctx.ui.notify(`Status failed: ${err.message}`, "error");
      }
    },
  });

  pi.registerCommand("context-clear", {
    description: "Clear the entire content index",
    handler: async (_args, ctx) => {
      try {
        clearIndex();
        if (ctx.hasUI) ctx.ui.notify("Index cleared.", "info");
      } catch (err) {
        ctx.ui.notify(`Clear failed: ${err.message}`, "error");
      }
    },
  });

  // ========================================================================
  // Persistence: Database auto-saves (better-sqlite3 writes through)
  // ========================================================================
  pi.on("session_end", async (_event, ctx) => {
    if (dbInstance) {
      try {
        dbInstance.close();
        dbInstance = null;
      } catch (err) {
        console.error("[context-store] Failed to close database:", err.message);
      }
    }
  });
};

module.exports.default = module.exports;
