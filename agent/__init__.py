"""
Файлы проекта:
- config: конфигурация и фабрика LLM.
- models: Pydantic-модели состояния и ответов.
- prompts: шаблоны промптов.
- chains: сборка цепочек LLM.
- handlers: ноды графа.
- graph: сборка графа LangGraph.
- cli: точка входа CLI.
"""

__all__ = [
    "config",
    "models",
    "prompts",
    "chains",
    "handlers",
    "graph",
    "cli",
]
