"""AniDaily MCP Server - 动画条漫生成工具集。

Tools:
    1. detect_and_stylize - 人脸检测 + 角色风格化
    2. edit_asset - 编辑已有素材
    3. analyze_scene - 场景分析
    4. generate_panel - 生成单格条漫
    5. script_editor - 剧本 md 读写
"""

import logging

from mcp.server.fastmcp import FastMCP

from src.mcp_tools.detect_stylize import register as register_detect_stylize
from src.mcp_tools.edit_asset import register as register_edit_asset
from src.mcp_tools.analyze_scene import register as register_analyze_scene
from src.mcp_tools.generate_panel import register as register_generate_panel
from src.mcp_tools.script_editor import register as register_script_editor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_server() -> FastMCP:
    """创建并配置 MCP Server。"""
    mcp = FastMCP("anidaily")

    register_detect_stylize(mcp)
    register_edit_asset(mcp)
    register_analyze_scene(mcp)
    register_generate_panel(mcp)
    register_script_editor(mcp)

    return mcp


def main() -> None:
    """启动 MCP Server。"""
    mcp = create_server()
    logger.info("Starting AniDaily MCP Server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
