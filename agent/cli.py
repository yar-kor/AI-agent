from __future__ import annotations

import logging
import sys

from .graph import build_graph
from .models import AgentState

logger = logging.getLogger(__name__)

try:
    sys.stdout.reconfigure(encoding="utf-8")  
except Exception:  
    pass


def run_cli():
    """Простой CLI-цикл для интерактивной работы."""
    app = build_graph()
    print("AI агент с LangGraph. Выход: Ctrl+C или пустая строка.")
    while True:
        try:
            query = input("Введите запрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыходим.")
            break

        if not query:
            print("Пустой ввод. Выходим.")
            break

        state = AgentState(user_query=query, search_mode="deep")
        result = app.invoke(state)

        # Безопасно извлекаем ответ независимо от типа (AgentState или dict)
        final_answer = getattr(result, "final_answer", None)
        if final_answer is None and isinstance(result, dict):
            final_answer = result.get("final_answer") or result.get("intermediate_result")
        final_answer = final_answer or "Ответ не сформирован из-за ошибки."

        print("\nФинальный ответ:")
        print(final_answer)
        print("-" * 40)
