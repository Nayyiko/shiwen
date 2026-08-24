"""S6 新裁角色扮演：1v1 与先贤对话，沉浸式叙事。

与 S4 辩论图的区别：
- S4 辩论：多智能体围绕辩题轮流发言，仲裁 + 漂移监测驱动
- S6 角色扮演：用户指定一位先贤，多轮对话，保持人设连贯性
"""

from .graph import build_roleplay_graph, RoleplayState

__all__ = ["build_roleplay_graph", "RoleplayState"]