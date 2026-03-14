"""API Key 轮询管理 & GenAI Client 工厂。

环境变量（通过 .env 加载）：
- API_KEYS: 逗号分隔的 API Key 列表
- BASE_URL: API 代理地址，默认 https://yunwu.ai/v1
"""

import itertools
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

# 从项目根目录加载 .env
_env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_env_path)

# --- API Key round-robin ---

_raw_keys = os.getenv("API_KEYS", "")
_API_KEYS: list[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not _API_KEYS:
    raise RuntimeError("API_KEYS not set in .env")

BASE_URL = os.getenv("BASE_URL", "https://yunwu.ai/v1")

_key_cycle = itertools.cycle(range(len(_API_KEYS)))
_key_lock = threading.Lock()
_key_failures: dict[int, int] = {}
_MAX_CONSECUTIVE_FAILURES = 3


def _next_key_index() -> int:
    """Get next healthy key index via round-robin, skipping temporarily failed keys."""
    with _key_lock:
        for _ in range(len(_API_KEYS)):
            idx = next(_key_cycle)
            if _key_failures.get(idx, 0) < _MAX_CONSECUTIVE_FAILURES:
                return idx
        _key_failures.clear()
        return next(_key_cycle)


def get_api_key() -> str:
    """Get next API key (round-robin with failover)."""
    return _API_KEYS[_next_key_index()]


def mark_key_success(key: str) -> None:
    """Reset failure count for a key on success."""
    with _key_lock:
        for i, k in enumerate(_API_KEYS):
            if k == key:
                _key_failures.pop(i, None)
                break


def mark_key_failure(key: str) -> None:
    """Increment failure count for a key on network error."""
    with _key_lock:
        for i, k in enumerate(_API_KEYS):
            if k == key:
                _key_failures[i] = _key_failures.get(i, 0) + 1
                break


def get_key_count() -> int:
    """Return the number of available API keys."""
    return len(_API_KEYS)


def get_api_base() -> str:
    """返回去掉 /v1 后缀的 API base URL。"""
    return BASE_URL.rstrip("/").replace("/v1", "")


def get_genai_client(timeout: int | None = None) -> "genai.Client":
    """创建 genai.Client，使用轮询 API key。"""
    from google import genai
    from google.genai.types import HttpOptions

    kwargs: dict = {"base_url": get_api_base()}
    if timeout is not None:
        kwargs["timeout"] = timeout
    http_opts = HttpOptions(**kwargs)
    return genai.Client(api_key=get_api_key(), http_options=http_opts)
