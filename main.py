from __future__ import annotations

"""
Точка входа для AI-агента на LangGraph.

Запуск:
1) Установите зависимости: pip install -r requirements.txt
2) Создайте .env с ключами провайдеров (минимум CLOUDRU_API_KEY для GigaChat; опционально GROQ_API_KEY и OPENROUTER_API_KEY) и при необходимости базовые URL.
3) При желании отредактируйте llm_config.yaml.
4) Выполните: python main.py
"""

import logging

from agent.cli import run_cli

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


if __name__ == "__main__":
    run_cli()
    
