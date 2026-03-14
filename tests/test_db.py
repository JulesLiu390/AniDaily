"""Tests for ConversationDB (SQLite persistence layer)."""

import json
import tempfile
from pathlib import Path

import pytest

from src.web.db import ConversationDB


@pytest.fixture()
def db(tmp_path: Path) -> ConversationDB:
    """Create a fresh in-tmp-dir database for each test."""
    db = ConversationDB(tmp_path / "test.db")
    yield db
    db.close()


# ---- Conversations CRUD ----


class TestConversations:
    def test_create_and_list(self, db: ConversationDB) -> None:
        cid = db.create_conversation("demo", "en")
        convs = db.list_conversations("demo")
        assert len(convs) == 1
        assert convs[0]["id"] == cid
        assert convs[0]["project"] == "demo"
        assert convs[0]["lang"] == "en"
        assert convs[0]["message_count"] == 0

    def test_list_filters_by_project(self, db: ConversationDB) -> None:
        db.create_conversation("alpha")
        db.create_conversation("beta")
        assert len(db.list_conversations("alpha")) == 1
        assert len(db.list_conversations("beta")) == 1
        assert len(db.list_conversations("gamma")) == 0

    def test_delete_conversation_cascade(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        db.append_history(cid, 0, {"role": "user", "parts": [{"text": "hi"}]})
        db.save_ui_message(cid, {"turnId": "t1", "role": "user", "content": "hi"})

        # Verify data exists before delete
        assert len(db.get_history(cid)) == 1
        assert len(db.get_ui_messages(cid)) == 1

        db.delete_conversation(cid)

        # Conversation gone
        assert len(db.list_conversations("proj")) == 0
        # CASCADE: children also gone
        assert len(db.get_history(cid)) == 0
        assert len(db.get_ui_messages(cid)) == 0

    def test_update_conversation_title(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        db.update_conversation(cid, title="My Comic")
        convs = db.list_conversations("proj")
        assert convs[0]["title"] == "My Comic"

    def test_update_conversation_lang(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj", lang="zh")
        db.update_conversation(cid, lang="en")
        convs = db.list_conversations("proj")
        assert convs[0]["lang"] == "en"

    def test_update_ignores_unknown_fields(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        # Should not raise
        db.update_conversation(cid, unknown_field="value")


# ---- History entries ----


class TestHistory:
    def test_append_and_get_ordered_by_seq(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        entry0 = {"role": "user", "parts": [{"text": "hello"}]}
        entry1 = {"role": "model", "parts": [{"text": "hi there"}]}
        # Insert out of order to verify seq-based ordering
        db.append_history(cid, 1, entry1)
        db.append_history(cid, 0, entry0)

        history = db.get_history(cid)
        assert len(history) == 2
        assert history[0] == entry0
        assert history[1] == entry1

    def test_append_history_updates_conversation_timestamp(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        convs_before = db.list_conversations("proj")
        old_updated = convs_before[0]["updated_at"]

        db.append_history(cid, 0, {"role": "user", "parts": [{"text": "x"}]})
        convs_after = db.list_conversations("proj")
        new_updated = convs_after[0]["updated_at"]
        assert new_updated >= old_updated

    def test_get_history_empty(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        assert db.get_history(cid) == []


# ---- UI messages ----


class TestUIMessages:
    def test_save_and_get_basic(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        db.save_ui_message(cid, {"turnId": "t1", "role": "user", "content": "hello"})
        msgs = db.get_ui_messages(cid)
        assert len(msgs) == 1
        assert msgs[0]["turnId"] == "t1"
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    def test_json_fields_parsed(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        tool_calls = [{"name": "detect_faces", "args": {"path": "/img.png"}}]
        images = ["/files/output/face1.png"]
        attached = ["/files/input/photo.jpg"]
        db.save_ui_message(
            cid,
            {
                "turnId": "t2",
                "role": "assistant",
                "content": "Found 1 face.",
                "toolCalls": tool_calls,
                "images": images,
                "attachedImages": attached,
                "type": "tool_result",
            },
        )
        msgs = db.get_ui_messages(cid)
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg["toolCalls"] == tool_calls
        assert msg["images"] == images
        assert msg["attachedImages"] == attached
        assert msg["type"] == "tool_result"

    def test_none_json_fields(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        db.save_ui_message(cid, {"turnId": "t3", "role": "user", "content": "hi"})
        msgs = db.get_ui_messages(cid)
        assert msgs[0]["toolCalls"] is None
        assert msgs[0]["images"] is None
        assert msgs[0]["attachedImages"] is None

    def test_message_count_in_list(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        db.save_ui_message(cid, {"turnId": "t1", "role": "user", "content": "a"})
        db.save_ui_message(cid, {"turnId": "t2", "role": "assistant", "content": "b"})
        db.save_ui_message(cid, {"turnId": "t3", "role": "user", "content": "c"})

        convs = db.list_conversations("proj")
        assert convs[0]["message_count"] == 3

    def test_ordering_by_id(self, db: ConversationDB) -> None:
        cid = db.create_conversation("proj")
        for i in range(5):
            db.save_ui_message(cid, {"turnId": f"t{i}", "role": "user", "content": f"msg {i}"})
        msgs = db.get_ui_messages(cid)
        assert [m["content"] for m in msgs] == [f"msg {i}" for i in range(5)]
