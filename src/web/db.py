"""SQLite conversation persistence for AniDaily."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class ConversationDB:
    """Manages conversation, history, and UI message persistence in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Enable WAL mode for concurrent reads
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                project     TEXT NOT NULL,
                lang        TEXT DEFAULT 'zh',
                title       TEXT DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS history_entries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                seq             INTEGER NOT NULL,
                content_json    TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ui_messages (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id     TEXT NOT NULL,
                turn_id             TEXT,
                role                TEXT,
                content             TEXT DEFAULT '',
                tool_calls_json     TEXT,
                images_json         TEXT,
                attached_images_json TEXT,
                message_type        TEXT,
                created_at          TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_history_conversation
                ON history_entries(conversation_id);

            CREATE INDEX IF NOT EXISTS idx_ui_messages_conversation
                ON ui_messages(conversation_id);
            """
        )
        self.conn.commit()

    # ---- Conversations CRUD ----

    def create_conversation(self, project: str, lang: str = "zh") -> str:
        """Create a new conversation and return its UUID."""
        cid = str(uuid.uuid4())
        now = _now()
        self.conn.execute(
            "INSERT INTO conversations (id, project, lang, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (cid, project, lang, now, now),
        )
        self.conn.commit()
        return cid

    def list_conversations(self, project: str) -> list[dict]:
        """List conversations for a project, ordered by updated_at DESC.

        Each dict includes a ``message_count`` field with the number of UI
        messages associated with that conversation.
        """
        rows = self.conn.execute(
            """
            SELECT c.id, c.project, c.lang, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN ui_messages m ON m.conversation_id = c.id
            WHERE c.project = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """,
            (project,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and CASCADE-delete its history and ui_messages."""
        self.conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self.conn.commit()

    def update_conversation(self, conversation_id: str, **kwargs: str) -> None:
        """Update mutable fields (title, lang) on a conversation."""
        allowed = {"title", "lang"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = _now()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [conversation_id]
        self.conn.execute(
            f"UPDATE conversations SET {set_clause} WHERE id = ?",  # noqa: S608
            values,
        )
        self.conn.commit()

    # ---- History (Gemini Content dicts) ----

    def append_history(self, conversation_id: str, seq: int, content_dict: dict) -> None:
        """Append a Gemini Content dict to the conversation history."""
        now = _now()
        self.conn.execute(
            "INSERT INTO history_entries (conversation_id, seq, content_json, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, seq, json.dumps(content_dict, ensure_ascii=False), now),
        )
        # Touch the conversation's updated_at timestamp
        self.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        self.conn.commit()

    def get_history(self, conversation_id: str) -> list[dict]:
        """Return history entries ordered by seq."""
        rows = self.conn.execute(
            "SELECT content_json FROM history_entries WHERE conversation_id = ? ORDER BY seq",
            (conversation_id,),
        ).fetchall()
        return [json.loads(r["content_json"]) for r in rows]

    # ---- UI Messages ----

    def replace_ui_messages(self, conversation_id: str, messages: list[dict]) -> None:
        """Replace all UI messages for a conversation (clear + re-insert)."""
        self.conn.execute("DELETE FROM ui_messages WHERE conversation_id = ?", (conversation_id,))
        for msg in messages:
            self._insert_ui_message(conversation_id, msg)
        self.conn.commit()

    def save_ui_message(self, conversation_id: str, msg: dict) -> None:
        """Persist a single UI message dict."""
        self._insert_ui_message(conversation_id, msg)
        self.conn.commit()

    def _insert_ui_message(self, conversation_id: str, msg: dict) -> None:
        now = _now()
        self.conn.execute(
            """
            INSERT INTO ui_messages
                (conversation_id, turn_id, role, content, tool_calls_json,
                 images_json, attached_images_json, message_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                msg.get("turnId"),
                msg.get("role"),
                msg.get("content", ""),
                json.dumps(msg["toolCalls"], ensure_ascii=False) if msg.get("toolCalls") else None,
                json.dumps(msg["images"], ensure_ascii=False) if msg.get("images") else None,
                json.dumps(msg["attachedImages"], ensure_ascii=False) if msg.get("attachedImages") else None,
                msg.get("type"),
                now,
            ),
        )

    def get_ui_messages(self, conversation_id: str) -> list[dict]:
        """Return UI messages for a conversation, ordered by id."""
        rows = self.conn.execute(
            """
            SELECT turn_id, role, content, tool_calls_json,
                   images_json, attached_images_json, message_type
            FROM ui_messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
        messages = []
        for r in rows:
            msg: dict = {
                "turnId": r["turn_id"],
                "role": r["role"],
                "content": r["content"] or "",
                "toolCalls": json.loads(r["tool_calls_json"]) if r["tool_calls_json"] else None,
                "images": json.loads(r["images_json"]) if r["images_json"] else None,
                "attachedImages": json.loads(r["attached_images_json"]) if r["attached_images_json"] else None,
                "type": r["message_type"],
            }
            messages.append(msg)
        return messages

    # ---- Lifecycle ----

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
