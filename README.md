<img width="1902" height="954" alt="image" src="https://github.com/user-attachments/assets/96931b25-e33e-4109-9440-0fe139b11951" />
<img width="1902" height="954" alt="image" src="https://github.com/user-attachments/assets/560163d3-a4a5-4fe4-a430-299a938c06e9" />



# pi-omlx-context-store

A Pi extension that stores large inputs in a local SQLite index with FTS5 full-text search, allowing the LLM to query and retrieve full chunk content or snippets. Keeps large imports out of the context window while enabling efficient retrieval.

## Problem

Large language models increasingly rely on website information, but face a critical limitation: context windows are too small to handle most inputs in their entirety. When large files, documents, or codebases are imported into a Pi session, they consume precious context window space, leaving less room for the actual conversation and reasoning.

This extension solves that by:
- **Intercepting large inputs** (>5KB) before they enter the context window
- **Storing them in a local SQLite database** with full-text search (FTS5)
- **Returning full chunk content** when the LLM queries by UID, or relevant snippets for general search
- **Significantly reduces context window usage** by keeping large imports out of the token window while preserving full retrievability

## Architecture

```
Large Input → Chunked (5KB chunks) → Stored in SQLite (FTS5 index)
                                              ↓
LLM queries by UID → Full chunk content returned
LLM searches by text → Relevant snippets returned
```

**Key design principle:** The extension stores full content in the database. For UID-based lookups, it returns the complete chunk body so the LLM can analyze full context. For general text searches, it returns ~500-character snippets around matching tokens — a lightweight pre-filter that still lets the LLM drill down into full chunks via UID when needed.

## Features

- **Local RAG**: Fast, deterministic, no API calls for search
- **FTS5 Full-Text Search**: Porter stemmer + Boolean queries (AND, OR, NOT, phrases)
- **UID-based retrieval**: Query by exact chunk UID to get full content
- **Smart chunking**: Splits on whitespace boundaries to avoid breaking words
- **Automatic interception**: Large inputs are transparently stored and transformed
- **SQLite persistence**: Data survives session restarts

## Installation

```bash
npm install better-sqlite3
```

Then load the extension:

```bash
pi --extension pi-omlx-context-store.js
```

Or register it in your Pi configuration.

## Usage

### Automatic (Large Inputs)

When a large input (>5KB) is detected, the extension:
1. Splits it into ~5KB chunks
2. Stores each chunk in the SQLite index
3. Transforms the context to instruct the LLM to query chunks by UID

The LLM receives instructions like:
```
[SYSTEM: Large input detected and stored in local RAG index. 
DO NOT process raw content — it has been removed from your context.
CRITICAL: All 2 chunk(s) are fully stored and indexed. ONLY NOW may you query.]
```

### Manual Commands

| Command | Description |
|---------|-------------|
| `/context-search <query>` | Search the index and view results |
| `/context-status` | Show index status (document count, total size) |
| `/context-clear` | Clear the entire content index |

### Tools

The extension registers these tools for LLM use:

| Tool | Description |
|------|-------------|
| `context_search` | Search the index. UID queries return full chunk content; text queries return snippets |
| `context_list` | List all documents with metadata |
| `context_stats` | Get index statistics (document count, total size) |
| `context_store` | Store text content in the index for later retrieval |
| `context_clear` | Clear the entire index |

### Database Schema

The `content` table stores:

| Column | Type | Purpose |
|--------|------|---------|
| `uid` | TEXT PRIMARY KEY | Unique identifier (e.g., `doc_1_chunk_0`) |
| `rid` | INTEGER | Row identifier mapped to FTS5 rowid |
| `title` | TEXT | Document title (first 150 chars of body if not provided) |
| `type` | TEXT | Inferred type: `code`, `data`, `document`, `style`, `markup` |
| `sz` | INTEGER | Body size in characters |
| `body` | TEXT | Full text content (stored in both main table and FTS trigger sync) |
| `queries` | INTEGER | Number of times this document was retrieved |
| `created_at` | INTEGER | Unix timestamp of creation |
| `session_id` | TEXT | Session that stored this document |
| `confirmed` | INTEGER | Confirmation flag (default 0) |

### Query Examples

**UID lookup (returns full chunk):**
```
uid:doc_1_chunk_0
doc_123_chunk_0
```

**Full-text search (returns snippets):**
```
"exact phrase"
keyword1 AND keyword2
"error handling" OR "exception"
```

## Database

- **Location**: `~/.pi/agent/context-store/pi-context-store.db`
- **Engine**: SQLite with FTS5 full-text search
- **Mode**: WAL (write-ahead logging for concurrent reads)
- **Tokenization**: Porter stemmer (matches "running" for query "run")

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CHUNK_SIZE` | `5000` | Characters per chunk (also the trigger threshold) |
| `MAX_QUERY_RESULTS` | `20` | Max results for FTS/LIKE queries |
| `CONTEXT_WINDOW_TAILROOM` | `8000` | Reserved tokens for LLM response |

## File Structure

```
pi-omlx-context-store/
├── index.js          # Extension source (SQLite + FTS5 storage layer)
├── package.json      # Package metadata and dependencies
└── .gitignore       # Excludes node_modules/ and *.db* files
```

## Dependencies

- `better-sqlite3` ^12.0.0 — Native SQLite bindings with FTS5 support

## Tradeoffs & Design Decisions

### Why SQLite FTS5 instead of vector embeddings?

| Aspect | FTS5 (current) | Vector Search (alternative) |
|--------|---------------|---------------------------|
| **Speed** | Fast, deterministic | Slower (embedding generation) |
| **Cost** | Zero API calls | Requires embedding model/API |
| **Semantic search** | No (keyword-only) | Yes (meaning-based matching) |
| **Setup** | Native to SQLite | Requires embedding model + vector store |
| **Accuracy** | Exact keyword matching | Fuzzy semantic similarity |

**Decision**: Chose FTS5 for speed, zero cost, and simplicity. The LLM can handle semantic reasoning itself once it has the full chunk content. This is a "retrieval then reason" approach rather than "retrieve semantically."

### Why return full chunk content for UID queries?

The extension returns the **complete body text** when querying by UID, not a snippet. This means:
- ✅ The LLM can analyze full context and determine relevance itself
- ✅ No information loss during retrieval
- ❌ More data enters the context window per query
- ❌ Irrelevant chunks waste tokens if queried unnecessarily

**Decision**: For UID queries, the extension returns full content and trusts the LLM to filter relevance. For general text searches, it returns lightweight snippets (~500 chars) as a convenience — the LLM can always drill into full chunks via UID.

### Why 5KB chunk size?

- Small enough to keep individual queries manageable in context
- Large enough to preserve meaningful context (code blocks, paragraphs)
- Splits on whitespace boundaries to avoid breaking words
- **Tradeoff**: Fixed size doesn't adapt to content type (code vs. prose vs. JSON)

### What's NOT implemented (future improvements)

| Feature | Benefit | Complexity |
|---------|---------|------------|
| **Semantic search** (embeddings) | Meaning-based retrieval beyond keywords | High — needs embedding model |
| **Content deduplication** | Avoid storing identical chunks twice | Low — hash-based detection |
| **Incremental updates** | Skip re-chunking unchanged content | Medium — diff-based detection |
| **Cross-session search** | Search across all sessions, not just current | Low — filter by session_id |
| **Query result caching** | Avoid re-fetching repeated queries | Low — LRU cache |
| **Compression** | Reduce disk usage for large bodies | Medium — compress/decompress overhead |
| **Metadata extraction** | Keywords, summaries for better search | Medium — NLP pipeline needed |
| **BM25 ranking** | Better result ordering than FTS5 default rank | Low — configurable in FTS5 |
| **TTL / expiration** | Auto-cleanup old session data | Low — timestamp-based purge |
| **Adaptive chunking** | Different sizes for code vs. prose | Medium — content-type detection |
| **Concurrent access** | Support multiple Pi instances safely | Low — file locking |

## License

MIT
