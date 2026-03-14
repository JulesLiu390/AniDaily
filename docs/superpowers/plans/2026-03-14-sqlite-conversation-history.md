# SQLite Conversation History Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist conversation history and UI messages to SQLite so conversations survive server restarts and can be listed/resumed in the frontend.

**Architecture:** Add a `db.py` module for SQLite CRUD, a `serializer.py` for Gemini Content/Part <-> JSON conversion, wire persistence into the existing Agent and API layer, and add a conversation list + restore flow in the frontend.

**Tech Stack:** Python `sqlite3` (stdlib), existing FastAPI + React stack. No new dependencies.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/web/db.py` | **Create** | SQLite connection, schema init, all CRUD operations |
| `src/web/serializer.py` | **Create** | Gemini `Content`/`Part` <-> JSON dict conversion |
| `src/web/agent.py` | **Modify** | Accept optional `history` in constructor; no DB calls inside Agent |
| `src/web/api.py` | **Modify** | Call `db.py` to save/restore history; add conversation list/delete endpoints |
| `web/frontend/src/api.ts` | **Modify** | Add `fetchConversations`, `deleteConversation`, `fetchMessages` API functions |
| `web/frontend/src/components/ChatPanel.tsx` | **Modify** | Load conversation list, restore messages on select, save UI messages |
| `tests/test_serializer.py` | **Create** | Unit tests for Content/Part serialization round-trip |
| `tests/test_db.py` | **Create** | Unit tests for DB CRUD operations |

---

## Chunk 1: Backend — Serializer + DB + Integration

### Task 1: Content/Part Serializer

**Files:**
- Create: `src/web/serializer.py`
- Create: `tests/test_serializer.py`

- [ ] **Step 1: Write the failing test for text Content round-trip**

```python
# tests/test_serializer.py
from google.genai.types import Content, Part

from src.web.serializer import content_to_dict, dict_to_content


def test_text_content_roundtrip():
    original = Content(role="user", parts=[Part.from_text(text="hello world")])
    d = content_to_dict(original)
    assert d["role"] == "user"
    assert d["parts"][0] == {"type": "text", "text": "hello world"}
    restored = dict_to_content(d)
    assert restored.role == "user"
    assert restored.parts[0].text == "hello world"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jules/Documents/Projects/AniDaily && uv run pytest tests/test_serializer.py::test_text_content_roundtrip -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write the serializer**

```python
# src/web/serializer.py
"""Gemini Content/Part <-> JSON serialization.

Handles 4 Part types:
- text: {"type": "text", "text": "..."}
- image: {"type": "image", "path": "/abs/path.png", "mime_type": "image/png"}
  (stores file path, NOT raw bytes — bytes are read from disk on restore)
- function_call: {"type": "function_call", "name": "...", "args": {...}}
- function_response: {"type": "function_response", "name": "...", "response": {...}}
"""

from pathlib import Path

from google.genai.types import Content, FunctionCall, FunctionResponse, Part


def part_to_dict(part: Part) -> dict | None:
    """Convert a single Part to a JSON-serializable dict."""
    if part.text is not None:
        return {"type": "text", "text": part.text}
    if part.function_call is not None:
        fc = part.function_call
        return {
            "type": "function_call",
            "name": fc.name,
            "args": dict(fc.args) if fc.args else {},
        }
    if part.function_response is not None:
        fr = part.function_response
        return {
            "type": "function_response",
            "name": fr.name,
            "response": dict(fr.response) if fr.response else {},
        }
    if part.inline_data is not None:
        # We don't store raw bytes — caller should have injected a path marker
        # before the image part: Part.from_text("[图片路径: /path/to/img.png]")
        return None
    return None


def dict_to_part(d: dict) -> Part | None:
    """Restore a Part from a dict. Returns None for unrestorable parts."""
    t = d.get("type")
    if t == "text":
        # If this is an image path marker, try to re-attach the image bytes
        text = d["text"]
        if text.startswith("[图片路径: ") and text.endswith("]"):
            img_path = Path(text[7:-1])  # extract path from marker
            if img_path.exists():
                mime = _guess_mime(img_path)
                return Part.from_bytes(data=img_path.read_bytes(), mime_type=mime)
            # File gone — keep as text marker so Gemini knows an image was here
        return Part.from_text(text=text)
    if t == "function_call":
        return Part(function_call=FunctionCall(name=d["name"], args=d.get("args", {})))
    if t == "function_response":
        return Part.from_function_response(name=d["name"], response=d.get("response", {}))
    return None


def content_to_dict(content: Content) -> dict:
    """Convert a Content object to a JSON-serializable dict."""
    parts = []
    for part in content.parts or []:
        d = part_to_dict(part)
        if d is not None:
            parts.append(d)
    return {"role": content.role, "parts": parts}


def dict_to_content(d: dict) -> Content:
    """Restore a Content object from a dict."""
    parts = []
    for pd in d.get("parts", []):
        part = dict_to_part(pd)
        if part is not None:
            parts.append(part)
    return Content(role=d["role"], parts=parts)


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/jules/Documents/Projects/AniDaily && uv run pytest tests/test_serializer.py::test_text_content_roundtrip -v`
Expected: PASS

- [ ] **Step 5: Write and run tests for function_call and function_response**

```python
# append to tests/test_serializer.py

def test_function_call_roundtrip():
    original = Content(
        role="model",
        parts=[
            Part.from_text(text="Let me detect faces."),
            Part(function_call=FunctionCall(
                name="detect_faces_in_image",
                args={"image_path": "/projects/test/input/photo.png"},
            )),
        ],
    )
    d = content_to_dict(original)
    assert len(d["parts"]) == 2
    assert d["parts"][1]["type"] == "function_call"
    assert d["parts"][1]["name"] == "detect_faces_in_image"

    restored = dict_to_content(d)
    assert restored.parts[0].text == "Let me detect faces."
    assert restored.parts[1].function_call.name == "detect_faces_in_image"


def test_function_response_roundtrip():
    original = Content(
        role="user",
        parts=[
            Part.from_function_response(
                name="detect_faces_in_image",
                response={"faces": [{"age": 25, "gender": "M"}]},
            ),
        ],
    )
    d = content_to_dict(original)
    assert d["parts"][0]["type"] == "function_response"

    restored = dict_to_content(d)
    assert restored.parts[0].function_response.name == "detect_faces_in_image"
```

Run: `cd /Users/jules/Documents/Projects/AniDaily && uv run pytest tests/test_serializer.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/web/serializer.py tests/test_serializer.py
git commit -m "feat: add Gemini Content/Part JSON serializer"
```

---

### Task 2: SQLite Database Module

**Files:**
- Create: `src/web/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the failing test for DB init and conversation CRUD**

```python
# tests/test_db.py
import tempfile
from pathlib import Path

from src.web.db import ConversationDB


def test_create_and_list_conversations():
    with tempfile.TemporaryDirectory() as tmp:
        db = ConversationDB(Path(tmp) / "test.db")

        cid = db.create_conversation("test_project", "zh")
        assert cid  # non-empty string

        convos = db.list_conversations("test_project")
        assert len(convos) == 1
        assert convos[0]["id"] == cid
        assert convos[0]["project"] == "test_project"


def test_delete_conversation():
    with tempfile.TemporaryDirectory() as tmp:
        db = ConversationDB(Path(tmp) / "test.db")
        cid = db.create_conversation("proj", "zh")
        db.delete_conversation(cid)
        assert db.list_conversations("proj") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jules/Documents/Projects/AniDaily && uv run pytest tests/test_db.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Write db.py with schema and conversation CRUD**

```python
# src/web/db.py
"""SQLite persistence for conversation history."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ConversationDB:
    """Thin wrapper around SQLite for conversation persistence."""

    def __init__(self, db_path: Path):
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                lang TEXT NOT NULL DEFAULT 'zh',
                title TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS history_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ui_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                turn_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                tool_calls_json TEXT,
                images_json TEXT,
                attached_images_json TEXT,
                message_type TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_history_conv ON history_entries(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_ui_conv ON ui_messages(conversation_id);
        """)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ---- Conversations ----

    def create_conversation(self, project: str, lang: str) -> str:
        cid = str(uuid.uuid4())
        now = self._now()
        self._conn.execute(
            "INSERT INTO conversations (id, project, lang, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (cid, project, lang, now, now),
        )
        self._conn.commit()
        return cid

    def list_conversations(self, project: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, project, lang, title, created_at, updated_at FROM conversations "
            "WHERE project = ? ORDER BY updated_at DESC",
            (project,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        self._conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self._conn.commit()

    def update_conversation(self, conversation_id: str, **kwargs) -> None:
        """Update title or lang. Only updates fields present in kwargs."""
        allowed = {"title", "lang"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = self._now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [conversation_id]
        self._conn.execute(
            f"UPDATE conversations SET {set_clause} WHERE id = ?", values
        )
        self._conn.commit()

    # ---- History (Gemini Content objects) ----

    def append_history(self, conversation_id: str, seq: int, content_dict: dict) -> None:
        """Append a serialized Content dict to history."""
        self._conn.execute(
            "INSERT INTO history_entries (conversation_id, seq, content_json, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, seq, json.dumps(content_dict, ensure_ascii=False), self._now()),
        )
        self._conn.commit()
        # Touch updated_at
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (self._now(), conversation_id),
        )
        self._conn.commit()

    def get_history(self, conversation_id: str) -> list[dict]:
        """Return all history entries ordered by seq."""
        rows = self._conn.execute(
            "SELECT content_json FROM history_entries WHERE conversation_id = ? ORDER BY seq",
            (conversation_id,),
        ).fetchall()
        return [json.loads(r["content_json"]) for r in rows]

    # ---- UI Messages ----

    def save_ui_message(self, conversation_id: str, msg: dict) -> None:
        """Save a single UI message dict."""
        self._conn.execute(
            "INSERT INTO ui_messages "
            "(conversation_id, turn_id, role, content, tool_calls_json, images_json, attached_images_json, message_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                msg.get("turnId", ""),
                msg.get("role", "user"),
                msg.get("content", ""),
                json.dumps(msg.get("toolCalls"), ensure_ascii=False) if msg.get("toolCalls") else None,
                json.dumps(msg.get("images"), ensure_ascii=False) if msg.get("images") else None,
                json.dumps(msg.get("attachedImages"), ensure_ascii=False) if msg.get("attachedImages") else None,
                msg.get("type"),
                self._now(),
            ),
        )
        self._conn.commit()

    def get_ui_messages(self, conversation_id: str) -> list[dict]:
        """Return all UI messages for a conversation."""
        rows = self._conn.execute(
            "SELECT turn_id, role, content, tool_calls_json, images_json, attached_images_json, message_type "
            "FROM ui_messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        result = []
        for r in rows:
            msg: dict = {
                "turnId": r["turn_id"],
                "role": r["role"],
                "content": r["content"],
            }
            if r["tool_calls_json"]:
                msg["toolCalls"] = json.loads(r["tool_calls_json"])
            if r["images_json"]:
                msg["images"] = json.loads(r["images_json"])
            if r["attached_images_json"]:
                msg["attachedImages"] = json.loads(r["attached_images_json"])
            if r["message_type"]:
                msg["type"] = r["message_type"]
            result.append(msg)
        return result

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jules/Documents/Projects/AniDaily && uv run pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Write and run tests for history and UI message persistence**

```python
# append to tests/test_db.py

def test_history_append_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        db = ConversationDB(Path(tmp) / "test.db")
        cid = db.create_conversation("proj", "zh")

        db.append_history(cid, 0, {"role": "user", "parts": [{"type": "text", "text": "hello"}]})
        db.append_history(cid, 1, {"role": "model", "parts": [{"type": "text", "text": "hi"}]})

        history = db.get_history(cid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["parts"][0]["text"] == "hi"


def test_ui_messages_save_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        db = ConversationDB(Path(tmp) / "test.db")
        cid = db.create_conversation("proj", "zh")

        db.save_ui_message(cid, {
            "turnId": "t1",
            "role": "user",
            "content": "hello",
            "type": "text",
        })
        db.save_ui_message(cid, {
            "turnId": "t1",
            "role": "assistant",
            "content": "hi there",
            "toolCalls": [{"tool": "detect_faces", "args": {}}],
        })

        msgs = db.get_ui_messages(cid)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "hello"
        assert msgs[1]["toolCalls"][0]["tool"] == "detect_faces"


def test_cascade_delete():
    with tempfile.TemporaryDirectory() as tmp:
        db = ConversationDB(Path(tmp) / "test.db")
        cid = db.create_conversation("proj", "zh")
        db.append_history(cid, 0, {"role": "user", "parts": []})
        db.save_ui_message(cid, {"turnId": "t1", "role": "user", "content": "hi"})

        db.delete_conversation(cid)
        assert db.get_history(cid) == []
        assert db.get_ui_messages(cid) == []
```

Run: `cd /Users/jules/Documents/Projects/AniDaily && uv run pytest tests/test_db.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/web/db.py tests/test_db.py
git commit -m "feat: add SQLite DB module for conversation persistence"
```

---

### Task 3: Wire DB into Agent + API

**Files:**
- Modify: `src/web/agent.py:859-866` (constructor)
- Modify: `src/web/api.py:270-284` (agent lifecycle), `src/web/api.py:320-346` (chat endpoint)

- [ ] **Step 1: Modify Agent constructor to accept pre-loaded history**

In `src/web/agent.py`, change the `__init__` method:

```python
# BEFORE (line 859-866):
def __init__(self, project_dir: Path | None = None, lang: str = "zh"):
    self.history: list[Content] = []
    ...

# AFTER:
def __init__(self, project_dir: Path | None = None, lang: str = "zh", history: list[Content] | None = None):
    self.history: list[Content] = history or []
    self.project_dir = project_dir
    self.lang = lang
    # Plan execution state
    self.active_plan: list[dict] | None = None
    self.plan_paused: bool = False
    self.plan_auto: bool = True
```

- [ ] **Step 2: Initialize global DB instance in api.py**

At the top of `src/web/api.py`, after the existing imports and logging setup, add:

```python
from src.web.db import ConversationDB

# ---- Database ----
_DB_PATH = PROJECT_ROOT / "anidaily.db"
db = ConversationDB(_DB_PATH)
logger.info(f"Database: {_DB_PATH}")
```

Also add `anidaily.db` to `.gitignore`.

- [ ] **Step 3: Rewrite `_get_agent` to restore from DB**

Replace `_get_agent` in `src/web/api.py`:

```python
from src.web.serializer import content_to_dict, dict_to_content

def _get_agent(conversation_id: str | None, project: str | None = None, lang: str = "zh") -> tuple[str, Agent]:
    """获取或创建对话 agent，优先从内存取，其次从 DB 恢复历史。"""
    # 1. In-memory hit
    if conversation_id and conversation_id in _agents:
        agent = _agents[conversation_id]
        agent.lang = lang
        return conversation_id, agent

    # 2. DB restore
    if conversation_id:
        history_dicts = db.get_history(conversation_id)
        if history_dicts:
            history = [dict_to_content(d) for d in history_dicts]
            project_dir = _project_path(project) if project else None
            agent = Agent(project_dir=project_dir, lang=lang, history=history)
            _agents[conversation_id] = agent
            return conversation_id, agent

    # 3. Brand new conversation
    project_name = project or "default"
    cid = db.create_conversation(project_name, lang)
    project_dir = _project_path(project) if project else None
    agent = Agent(project_dir=project_dir, lang=lang)
    _agents[cid] = agent
    return cid, agent
```

- [ ] **Step 4: Save history entries after each AI round**

In `src/web/api.py`, modify the `_stream_chat` function to persist history after each SSE stream completes. Wrap the `_run` inner function:

```python
async def _stream_chat(agent: Agent, cid: str, message: str, image_paths: list[str] | None) -> AsyncGenerator[str, None]:
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    history_len_before = len(agent.history)

    def _run():
        try:
            for event in agent.chat_stream(message, image_paths=image_paths):
                queue.put_nowait(event)
        except Exception as e:
            logger.exception("Agent stream error")
            queue.put_nowait({"event": "error", "message": str(e)})
        finally:
            # Persist new history entries to DB
            for i in range(history_len_before, len(agent.history)):
                try:
                    db.append_history(cid, i, content_to_dict(agent.history[i]))
                except Exception:
                    logger.exception("Failed to persist history entry")
            queue.put_nowait(None)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run)

    yield _sse_line("conversation_id", {"conversation_id": cid})

    while True:
        event = await queue.get()
        if event is None:
            break
        event_type = event.get("event", "unknown")
        yield _sse_line(event_type, event)
```

- [ ] **Step 5: Auto-generate conversation title from first user message**

In the `chat` endpoint in `src/web/api.py`, after `_get_agent`, add title generation for new conversations:

```python
@app.post("/api/chat")
async def chat(req: ChatRequest):
    cid, agent = _get_agent(req.conversation_id, req.project, req.lang or "zh")

    # ... existing plan_action handling ...

    # Auto-title on first message
    if not req.conversation_id and message:
        title = message[:50].strip()
        db.update_conversation(cid, title=title)

    return StreamingResponse(...)
```

- [ ] **Step 6: Add conversation list and delete endpoints**

Append to `src/web/api.py`:

```python
# ========== 对话历史 ==========

@app.get("/api/conversations")
def list_conversations(project: str) -> list[dict]:
    """列出项目的所有对话。"""
    return db.list_conversations(project)


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    """删除对话及其历史。"""
    _agents.pop(conversation_id, None)
    db.delete_conversation(conversation_id)
    return {"deleted": True}
```

- [ ] **Step 7: Add UI message save endpoint**

Append to `src/web/api.py`:

```python
class SaveMessagesRequest(BaseModel):
    conversation_id: str
    messages: list[dict]


@app.post("/api/conversations/{conversation_id}/messages")
def save_ui_messages(conversation_id: str, req: SaveMessagesRequest) -> dict:
    """保存前端 UI 消息（用于恢复聊天界面）。"""
    for msg in req.messages:
        db.save_ui_message(conversation_id, msg)
    return {"saved": len(req.messages)}


@app.get("/api/conversations/{conversation_id}/messages")
def get_ui_messages(conversation_id: str) -> list[dict]:
    """获取对话的 UI 消息。"""
    return db.get_ui_messages(conversation_id)
```

- [ ] **Step 8: Verify backend starts without errors**

Run: `cd /Users/jules/Documents/Projects/AniDaily && timeout 5 uv run uvicorn src.web.api:app --port 8000 2>&1 || true`
Expected: "Application startup complete" with no import errors; `anidaily.db` created in project root

- [ ] **Step 9: Commit**

```bash
git add src/web/agent.py src/web/api.py src/web/db.py src/web/serializer.py .gitignore
git commit -m "feat: wire SQLite persistence into agent and API layer"
```

---

## Chunk 2: Frontend — Conversation List + Restore

### Task 4: Frontend API Functions

**Files:**
- Modify: `web/frontend/src/api.ts`

- [ ] **Step 1: Add conversation API functions**

Append to `web/frontend/src/api.ts` before the `getFileUrl` function:

```typescript
// ========== 对话历史 ==========

export interface Conversation {
  id: string;
  project: string;
  lang: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export async function fetchConversations(project: string): Promise<Conversation[]> {
  const res = await fetch(`${API_BASE}/api/conversations?project=${encodeURIComponent(project)}`);
  return res.json();
}

export async function deleteConversation(conversationId: string): Promise<{ deleted: boolean }> {
  const res = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(conversationId)}`, {
    method: "DELETE",
  });
  return res.json();
}

export async function fetchUIMessages(conversationId: string): Promise<ChatMessage[]> {
  const res = await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(conversationId)}/messages`);
  return res.json();
}

export async function saveUIMessages(conversationId: string, messages: ChatMessage[]): Promise<void> {
  await fetch(`${API_BASE}/api/conversations/${encodeURIComponent(conversationId)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: conversationId, messages }),
  });
}
```

- [ ] **Step 2: Commit**

```bash
git add web/frontend/src/api.ts
git commit -m "feat: add conversation history API functions to frontend"
```

---

### Task 5: ChatPanel — Save and Restore Messages

**Files:**
- Modify: `web/frontend/src/components/ChatPanel.tsx`

- [ ] **Step 1: Add imports and conversation list state**

At the top of `ChatPanel.tsx`, add the new API imports:

```typescript
import { deleteAsset, getFileUrl, streamMessage, uploadImage,
         fetchConversations, fetchUIMessages, saveUIMessages, deleteConversation } from "../api";
import type { AttachedImage, CharacterOption, ChatMessage, Conversation, ImageVerdict, PlanStep, ToolCallInfo } from "../api";
```

Add conversation list state after the existing state declarations (around line 30):

```typescript
const [conversations, setConversations] = useState<Conversation[]>([]);
const [showHistory, setShowHistory] = useState(false);
```

- [ ] **Step 2: Load conversation list on project change**

Modify the existing `useEffect` that resets on project change (around line 44):

```typescript
useEffect(() => {
  setMessages([]);
  setConversationId(null);
  setInput("");
  setAttachedImages([]);
  setEditingTurnId(null);
  setPlanSteps([]);
  setPlanPaused(false);
  setPlanTurnId(null);
  // Load conversation history list
  fetchConversations(project).then(setConversations).catch(() => {});
}, [project]);
```

- [ ] **Step 3: Save UI messages after each assistant response completes**

In the `makeCallbacks` function, in the `onDone` callback, add a save call. Find the existing `onDone` handler and add:

```typescript
onDone: () => {
  setLoading(false);
  onNewImages?.();
  // Save UI messages to DB
  if (conversationIdRef.current) {
    // Use a ref to get the latest messages
    setMessages((prev) => {
      // Find messages not yet saved (from this turn)
      const unsaved = prev.filter(m => m.turnId === currentTurnIdRef.current);
      if (unsaved.length > 0 && conversationIdRef.current) {
        saveUIMessages(conversationIdRef.current, unsaved).catch(() => {});
      }
      return prev;
    });
  }
},
```

Note: You'll need to add a `conversationIdRef` to track the current conversation ID. Add near the other refs:

```typescript
const conversationIdRef = useRef<string | null>(null);
// Keep ref in sync with state
useEffect(() => { conversationIdRef.current = conversationId; }, [conversationId]);
```

And a `currentTurnIdRef` to track the current assistant turn:

```typescript
const currentTurnIdRef = useRef<string | null>(null);
```

Set it when creating a new assistant turn in the callbacks.

- [ ] **Step 4: Add conversation restore function**

Add this function inside the ChatPanel component:

```typescript
const handleRestoreConversation = async (conv: Conversation) => {
  try {
    const msgs = await fetchUIMessages(conv.id);
    setMessages(msgs);
    setConversationId(conv.id);
    setShowHistory(false);
    setPlanSteps([]);
    setPlanPaused(false);
  } catch (err) {
    console.error("Failed to restore conversation:", err);
  }
};
```

- [ ] **Step 5: Add conversation history UI**

Add a "History" button and dropdown near the top of the chat panel JSX. Find the chat header area (typically at the top of the return JSX) and add:

```tsx
{/* Conversation history toggle */}
<div className="relative">
  <button
    onClick={() => { setShowHistory(!showHistory); if (!showHistory) fetchConversations(project).then(setConversations); }}
    className="text-sm text-gray-500 hover:text-gray-700 px-2 py-1 rounded hover:bg-gray-100"
    title={t("chat.history")}
  >
    {t("chat.history")}
  </button>
  {showHistory && (
    <div className="absolute left-0 top-full mt-1 w-72 bg-white border rounded-lg shadow-lg z-50 max-h-64 overflow-y-auto">
      {conversations.length === 0 ? (
        <div className="p-3 text-sm text-gray-400">{t("chat.noHistory")}</div>
      ) : (
        conversations.map((conv) => (
          <div
            key={conv.id}
            className={`flex items-center justify-between p-2 hover:bg-gray-50 cursor-pointer border-b last:border-0 ${conv.id === conversationId ? "bg-blue-50" : ""}`}
          >
            <div className="flex-1 min-w-0" onClick={() => handleRestoreConversation(conv)}>
              <div className="text-sm font-medium truncate">{conv.title || t("chat.untitled")}</div>
              <div className="text-xs text-gray-400">{new Date(conv.updated_at).toLocaleString()}</div>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id).then(() => setConversations(cs => cs.filter(c => c.id !== conv.id))); }}
              className="text-xs text-red-400 hover:text-red-600 ml-2 shrink-0"
            >
              {t("chat.delete")}
            </button>
          </div>
        ))
      )}
    </div>
  )}
</div>
```

- [ ] **Step 6: Add i18n entries**

Add the new translation keys to `web/frontend/src/i18n.ts`:

```typescript
// In the zh translations:
"chat.history": "历史记录",
"chat.noHistory": "暂无历史记录",
"chat.untitled": "未命名对话",
"chat.delete": "删除",

// In the en translations:
"chat.history": "History",
"chat.noHistory": "No conversation history",
"chat.untitled": "Untitled",
"chat.delete": "Delete",
```

- [ ] **Step 7: Add "New Chat" button**

Add a button to start a fresh conversation (clear messages + conversationId):

```tsx
<button
  onClick={() => { setMessages([]); setConversationId(null); setPlanSteps([]); setPlanPaused(false); setShowHistory(false); }}
  className="text-sm text-blue-500 hover:text-blue-700 px-2 py-1 rounded hover:bg-blue-50"
>
  {t("chat.newChat")}
</button>
```

i18n: `"chat.newChat": "新对话"` / `"chat.newChat": "New Chat"`

- [ ] **Step 8: Verify frontend compiles**

Run: `cd /Users/jules/Documents/Projects/AniDaily/web/frontend && npm run build`
Expected: Build succeeds with no TypeScript errors

- [ ] **Step 9: Commit**

```bash
git add web/frontend/src/components/ChatPanel.tsx web/frontend/src/api.ts web/frontend/src/i18n.ts
git commit -m "feat: add conversation history list and restore in frontend"
```

---

### Task 6: Manual End-to-End Verification

- [ ] **Step 1: Start backend**

Run: `cd /Users/jules/Documents/Projects/AniDaily && uv run uvicorn src.web.api:app --reload --port 8000`
Verify: `anidaily.db` file created, "Database:" line in log

- [ ] **Step 2: Start frontend**

Run: `cd /Users/jules/Documents/Projects/AniDaily/web/frontend && npm run dev`

- [ ] **Step 3: Test conversation persistence flow**

1. Open browser, select a project
2. Send a message → verify conversation_id appears in SSE
3. Send a few messages, verify responses
4. Restart the backend (`Ctrl+C` then re-run uvicorn)
5. Click "History" button → should see previous conversation listed with title
6. Click on it → messages should restore
7. Send a new message → agent should have full context from restored history
8. Delete a conversation → verify it disappears from list

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete SQLite conversation history with frontend UI"
```
