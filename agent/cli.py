from __future__ import annotations

import logging
import sys
import os

from .graph import build_graph
from .config import get_llm_config
from .models import AgentState

logger = logging.getLogger(__name__)

try:
    sys.stdout.reconfigure(encoding="utf-8")  
except Exception:  
    pass


def run_cli():
 
    app = build_graph()
    print("AI агент запущен. Выход: Ctrl+C или пустая строка.")
    print("Команды: model=<имя> [запрос] — выбрать модель; models — показать список моделей.")
    while True:
        try:
            query = input("Введите запрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыходим.")
            break

        if not query:
            print("Пустой ввод. Выходим.")
            break

        if query.lower() == "models":
            cfg = get_llm_config()
            nodes = cfg.get("nodes", {})
            seen = []
            for data in nodes.values():
                for m in data.get("models", []):
                    name = m.get("model")
                    if name and name not in seen:
                        seen.append(name)
            print("Доступные модели:")
            for name in seen:
                print(f"- {name}")
            continue

        if query.lower().startswith("model="):
            raw = query.split("=", 1)[1].strip()
            parts = raw.split(maxsplit=1)
            value = parts[0] if parts else ""
            tail = parts[1] if len(parts) > 1 else ""
            if value:
                os.environ["LLM_PREFERRED_MODEL"] = value
                print(f"Выбрана prefer-модель: {value}")
            else:
                os.environ.pop("LLM_PREFERRED_MODEL", None)
                print("Prefer-модель сброшена (используется порядок из конфигурации).")
            if not tail:
                continue
            query = tail

        state = AgentState(user_query=query, search_mode="deep")
        result = app.invoke(state)

        final_answer = getattr(result, "final_answer", None)
        if final_answer is None and isinstance(result, dict):
            final_answer = result.get("final_answer") or result.get("intermediate_result")
        final_answer = final_answer or "Ответ не сформирован из-за ошибки."

        print("\nОтвет:")
        print(final_answer)
        print("-" * 40)
