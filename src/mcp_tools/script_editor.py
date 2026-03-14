"""Tool 5: 剧本 md 读写。

读取、写入、更新 markdown 剧本文件。
"""

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    """注册剧本编辑工具。"""

    @mcp.tool()
    def read_script(file_path: str) -> dict:
        """读取剧本文件内容。

        Args:
            file_path: md 文件路径。

        Returns:
            文件内容。
        """
        p = Path(file_path)
        if not p.exists():
            return {"error": f"文件不存在: {file_path}"}
        content = p.read_text(encoding="utf-8")
        return {
            "file_path": file_path,
            "content": content,
            "lines": len(content.splitlines()),
        }

    @mcp.tool()
    def write_script(file_path: str, content: str) -> dict:
        """写入/覆盖剧本文件。

        Args:
            file_path: md 文件路径。
            content: 完整文件内容。

        Returns:
            写入结果。
        """
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {
            "file_path": file_path,
            "lines": len(content.splitlines()),
            "message": "写入成功",
        }

    @mcp.tool()
    def update_script(
        file_path: str,
        old_text: str,
        new_text: str,
    ) -> dict:
        """替换剧本文件中的指定文本。

        Args:
            file_path: md 文件路径。
            old_text: 要替换的原文本。
            new_text: 替换后的新文本。

        Returns:
            替换结果。
        """
        p = Path(file_path)
        if not p.exists():
            return {"error": f"文件不存在: {file_path}"}

        content = p.read_text(encoding="utf-8")
        if old_text not in content:
            return {"error": f"未找到要替换的文本", "file_path": file_path}

        count = content.count(old_text)
        new_content = content.replace(old_text, new_text)
        p.write_text(new_content, encoding="utf-8")

        return {
            "file_path": file_path,
            "replacements": count,
            "message": f"替换了 {count} 处",
        }

    @mcp.tool()
    def search_script(
        file_path: str,
        keyword: str,
    ) -> dict:
        """在剧本文件中搜索关键词。

        Args:
            file_path: md 文件路径。
            keyword: 搜索关键词。

        Returns:
            匹配的行号和内容。
        """
        p = Path(file_path)
        if not p.exists():
            return {"error": f"文件不存在: {file_path}"}

        content = p.read_text(encoding="utf-8")
        matches = []
        for i, line in enumerate(content.splitlines(), 1):
            if keyword in line:
                matches.append({"line": i, "content": line.strip()})

        return {
            "file_path": file_path,
            "keyword": keyword,
            "matches": matches,
            "count": len(matches),
        }
