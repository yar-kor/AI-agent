from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Загружаем переменные из .env
load_dotenv()


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """
    Создаем и кэшируем клиент OpenAI-совместимого API (cloud.ru).

    Ожидаемые переменные:
    - API_KEY: ключ доступа
    - API_BASE (необязательно): базовый URL, по умолчанию https://foundation-models.api.cloud.ru/v1
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError(
            "Переменная окружения API_KEY не найдена. "
            "Добавьте ее в .env или окружение с вашим ключом cloud.ru."
        )

    base_url = os.getenv("API_BASE", "https://foundation-models.api.cloud.ru/v1")

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model="ai-sage/GigaChat3-10B-A1.8B",
        temperature=0,
        max_tokens=2048,
    )
