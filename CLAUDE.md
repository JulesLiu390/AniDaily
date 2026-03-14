# AniDaily

AI-powered vertical comic strip (条漫) generator: real photos → anime characters → manga panels.

## Quick Start

```bash
# Backend
uv run uvicorn src.web.api:app --reload --port 8000

# Frontend
cd web/frontend && npm run dev
```

## Architecture

- **Backend**: FastAPI + Gemini function calling (SSE streaming agent pattern)
- **Frontend**: React + TypeScript + Vite + Tailwind CSS v3
- **AI Models**: Gemini 3-flash (chat), 3.1-flash (image gen), 2.5-flash-lite (VLM)
- **Face Detection**: InsightFace (buffalo_l, local CPU)
- **API Proxy**: yunwu.ai with round-robin key rotation

## Project Structure

```
src/
  web/
    api.py          # FastAPI routes, SSE chat endpoint
    agent.py        # Gemini conversation agent, tool dispatch, DAG executor
  tools/            # Core tool implementations (face detection, stylization, image gen)
  mcp_tools/        # MCP tool definitions (parallel interface for external clients)
  mcp_server.py     # MCP server entry point (CLI: anidaily-mcp)
web/frontend/src/
  components/       # React components (ChatPanel, AssetSidebar, etc.)
  api.ts            # SSE parser, API client
projects/{name}/    # Per-project asset storage
  input/            # Uploaded photos
  output/           # Generated assets (stylized/, faces/, scenes/, panels/)
  scripts/          # Markdown comic scripts
  style.md          # Project style guide
```

## Key Conventions

- All asset directories use `assets.json` for metadata: `{filename: {name, description}}`
- Paths in the agent/frontend are relative to `projects/{name}/`; `_resolve_path()` converts to absolute
- SSE events: `text_delta`, `tool_start`, `tool_end`, `plan_gate`, `done`
- Complex tasks use DAG-based plans (`propose_plan`) with `depends_on` and `needs_confirm` gating
- Language: supports zh (default) and en via `LanguageContext`

## Environment

`.env` requires:
- `API_KEYS` — comma-separated Gemini proxy keys (round-robin with failover)
- `BASE_URL` — API proxy URL

## Testing

```bash
uv run pytest tests/
```
