# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

Requires Python 3.13+ and `uv`. On Windows, use Git Bash.

```bash
# Install dependencies
uv sync

# Set up environment (copy and fill in ANTHROPIC_API_KEY)
cp .env.example .env

# Start the server (from project root)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

App runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## Architecture

All backend code lives in `backend/` and is run from that directory (so imports are relative to it). The server serves the `frontend/` as static files at `/`.

### Request Flow

A user query goes through this chain:

```
app.py (FastAPI route)
  → rag_system.py (RAGSystem) — main orchestrator
      → session_manager.py   — fetches conversation history
      → ai_generator.py      — 1st Claude API call (with tools)
          → search_tools.py  — ToolManager executes the tool
              → vector_store.py — ChromaDB semantic search
          → ai_generator.py  — 2nd Claude API call (synthesizes answer)
      → session_manager.py   — stores exchange in history
```

Claude is given the `search_course_content` tool and decides autonomously whether to call it. If it does, the tool result is injected back and a second call produces the final answer.

### Key Architectural Decisions

**Two ChromaDB collections:** `course_catalog` stores course-level metadata (title, instructor, links, lessons JSON); `course_content` stores chunked lesson text. Course name resolution is semantic — `_resolve_course_name()` does a vector search on the catalog before filtering content, so partial/fuzzy course names work.

**Tool-as-search pattern:** Rather than always retrieving context before calling Claude, the system lets Claude decide when to search via Anthropic's tool-use API. `search_tools.py` defines the tool schema; `ai_generator.py` handles the two-turn tool loop; `rag_system.py` retrieves sources from `tool_manager.get_last_sources()` after the call.

**Session history is injected into the system prompt**, not the messages array. `SessionManager` keeps the last `MAX_HISTORY=2` exchanges (4 messages) and formats them as a plain string appended to the system prompt.

### Document Format

Course `.txt` files in `docs/` must follow this structure for `document_processor.py` to parse them correctly:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 2: <lesson title>
...
```

Chunks are sentence-split with size=800 chars, overlap=100 chars. The first chunk of each lesson is prefixed with `"Lesson N content: "` for retrieval context.

### Configuration

All tuneable parameters are in `backend/config.py` as a single `Config` dataclass: model name, embedding model (`all-MiniLM-L6-v2`), chunk size/overlap, max search results, history length, and ChromaDB path (`./chroma_db` relative to `backend/`).
