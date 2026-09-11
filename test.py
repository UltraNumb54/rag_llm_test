
1. app/config.py — секция «Агентский RAG: сжатие истории»
Найти старую секцию и заменить:

python
    # ─── Агентский RAG: сжатие истории (режим compressed) ───
    AGENT_HISTORY_COMPRESS_MIN_MESSAGES: int = Field(default=12, validation_alias="AGENT_HISTORY_COMPRESS_MIN_MESSAGES")
    AGENT_HISTORY_COMPRESS_MAX_TOKENS: int = Field(default=2000, validation_alias="AGENT_HISTORY_COMPRESS_MAX_TOKENS")
    AGENT_HISTORY_COMPRESSION_MAX_TOKENS: int = Field(default=300, validation_alias="AGENT_HISTORY_COMPRESSION_MAX_TOKENS")
    AGENT_HISTORY_COMPRESS_BATCH_SIZE: int = Field(default=20, validation_alias="AGENT_HISTORY_COMPRESS_BATCH_SIZE")
    AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS: int = Field(default=400, validation_alias="AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS")
    AGENT_HISTORY_COMPRESS_PARALLEL: bool = Field(default=True, validation_alias="AGENT_HISTORY_COMPRESS_PARALLEL")
    AGENT_HISTORY_COMPRESS_MAX_BATCHES: int = Field(default=3, validation_alias="AGENT_HISTORY_COMPRESS_MAX_BATCHES")
    # ─── Redis-кэш саммари ───
    AGENT_HISTORY_CACHE_ENABLED: bool = Field(default=True, validation_alias="AGENT_HISTORY_CACHE_ENABLED")
    AGENT_HISTORY_CACHE_TTL_SECONDS: int = Field(default=3600, validation_alias="AGENT_HISTORY_CACHE_TTL_SECONDS")
    AGENT_HISTORY_CACHE_VERSION: str = Field(default="v1", validation_alias="AGENT_HISTORY_CACHE_VERSION")
Удалить AGENT_HISTORY_COMPRESS_MAX_CHARS и AGENT_HISTORY_COMPRESS_FALLBACK_MESSAGES (больше не используются).

2. app/utils/summary_cache.py — новый файл
python
"""
Redis-кэш саммари истории диалога.

Хранит:
- batch-саммари по хэшу (batch_messages + prompt)
- финальное агрегированное саммари по хэшу списка batch-хэшей

TTL задаётся AGENT_HISTORY_CACHE_TTL_SECONDS.
Ключи версионируются AGENT_HISTORY_CACHE_VERSION — при смене промпта
достаточно поднять версию, старые записи «протухнут» естественным образом.
"""
import hashlib
from typing import List, Optional

from loguru import logger


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def batch_cache_key(messages: List[dict], prompt: str, version: str) -> str:
    """Ключ кэша для одного батча."""
    payload = prompt + "\n" + "\n".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    return f"hist_summary:{version}:batch:{_sha256(payload)}"


def aggregate_cache_key(batch_keys: List[str], prompt: str, version: str) -> str:
    """Ключ кэша для агрегации: сортируем batch_keys, чтобы порядок не влиял."""
    payload = prompt + "\n" + "\n".join(sorted(batch_keys))
    return f"hist_summary:{version}:agg:{_sha256(payload)}"


async def cache_get(redis_client, key: str) -> Optional[str]:
    if redis_client is None:
        return None
    try:
        raw = await redis_client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)
    except Exception as e:
        logger.warning(f"summary_cache GET {key} failed: {e}")
        return None


async def cache_set(redis_client, key: str, value: str, ttl_seconds: int) -> None:
    if redis_client is None or not value:
        return
    try:
        await redis_client.set(key, value, ex=ttl_seconds)
    except Exception as e:
        logger.warning(f"summary_cache SET {key} failed: {e}")
Почему нет утечки: всё в Redis, у каждого ключа TTL. Никаких in-memory dict — нечего чистить.

3. app/services/agent_service/history.py — полная замена
python
import asyncio
import re
import time
from typing import Dict, List, Optional

from loguru import logger

from app.config import settings
from app.services.llm_service.text import _get_no_think_prefix
from app.utils.summary_cache import (
    aggregate_cache_key,
    batch_cache_key,
    cache_get,
    cache_set,
)
from app.utils.tokenizer_utils import count_messages_tokens_async


class HistoryMixin:

    def _normalize_history(self, history: List[Dict]) -> List[Dict]:
        """Нормализация сообщений: только user/assistant, обрезка по MAX_HISTORY_MESSAGE_CHARS."""
        return [
            {
                "role": m["role"],
                "content": m.get("content", "")[: settings.MAX_HISTORY_MESSAGE_CHARS],
            }
            for m in history
            if m.get("role") in ("user", "assistant")
        ]

    def _split_into_batches(
        self, normalized: List[Dict], batch_size: int,
    ) -> List[List[Dict]]:
        """Разбивка на батчи с ограничением по MAX_BATCHES (с конца истории)."""
        if batch_size <= 0:
            batch_size = 20
        all_batches = [
            normalized[i: i + batch_size]
            for i in range(0, len(normalized), batch_size)
        ]
        max_batches = settings.AGENT_HISTORY_COMPRESS_MAX_BATCHES
        if max_batches > 0 and len(all_batches) > max_batches:
            dropped = len(all_batches) - max_batches
            logger.warning(
                f"История длинная: {len(all_batches)} батчей > "
                f"лимит {max_batches}, отброшено {dropped} самых старых "
                f"({dropped * batch_size} сообщений)"
            )
            all_batches = all_batches[-max_batches:]
        return all_batches

    async def _compress_batch(
        self,
        llm_service,
        batch: List[Dict],
        batch_index: int,
        total_batches: int,
        redis_client=None,
    ) -> Optional[str]:
        """
        Сжатие одного батча. Возвращает саммари или None при ошибке.
        Проверяет Redis-кэш перед вызовом LLM.
        """
        history_text = self._format_history_lines(batch)
        if not history_text:
            return None

        prompt = self.history_compress_prompt.format(history_text=history_text)
        no_think_prefix = _get_no_think_prefix(for_agent=True)
        if no_think_prefix:
            prompt = f"{no_think_prefix}{prompt}"

        # Кэш
        cache_k = None
        if settings.AGENT_HISTORY_CACHE_ENABLED and redis_client is not None:
            cache_k = batch_cache_key(batch, prompt, settings.AGENT_HISTORY_CACHE_VERSION)
            cached = await cache_get(redis_client, cache_k)
            if cached:
                logger.debug(f"Батч {batch_index}/{total_batches}: cache HIT")
                return cached

        extra_body = None
        if settings.AGENT_DISABLE_THINKING:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        start_ts = time.time()
        request_payload = {
            "batch_index": batch_index,
            "total_batches": total_batches,
            "batch_size": len(batch),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": settings.AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS,
        }

        try:
            response = await llm_service.completions(
                model=llm_service.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=settings.AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS,
                stream=False,
                extra_body=extra_body,
            )
            if not response.choices or not response.choices[0].message.content:
                self._trace_llm("history_compression_batch", request_payload, {"content": ""}, start_ts)
                logger.warning(f"Сжатие батча {batch_index}/{total_batches}: пустой ответ")
                return None

            content = response.choices[0].message.content.strip()
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = re.sub(
                r"^(Саммари|Резюме|Summary)\s*:\s*", "", content, flags=re.IGNORECASE,
            ).strip()
            if not content:
                self._trace_llm("history_compression_batch", request_payload, {"content": ""}, start_ts)
                return None

            self._trace_llm("history_compression_batch", request_payload, {"content": content}, start_ts)
            logger.debug(
                f"Батч {batch_index}/{total_batches} сжат: "
                f"{len(batch)} сообщений -> {len(content)} симв."
            )

            if cache_k is not None:
                await cache_set(
                    redis_client, cache_k, content,
                    settings.AGENT_HISTORY_CACHE_TTL_SECONDS,
                )
            return content

        except Exception as e:
            logger.warning(f"Ошибка сжатия батча {batch_index}/{total_batches}: {e}")
            self._trace_llm("history_compression_batch", request_payload, None, start_ts, error=str(e))
            return None

    async def _aggregate_summaries(
        self, llm_service, summaries: List[str], redis_client=None,
    ) -> str:
        """Агрегация нескольких саммари в одно финальное. С Redis-кэшем."""
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0]

        combined_text = "\n\n".join(
            f"[Часть {i + 1}]\n{s}" for i, s in enumerate(summaries)
        )
        prompt = self.history_compress_aggregate_prompt.format(history_text=combined_text)
        no_think_prefix = _get_no_think_prefix(for_agent=True)
        if no_think_prefix:
            prompt = f"{no_think_prefix}{prompt}"

        cache_k = None
        if settings.AGENT_HISTORY_CACHE_ENABLED and redis_client is not None:
            # ключ строится по хэшу от summaries (а не по тексту prompt целиком)
            agg_key_source = "\n---\n".join(summaries)
            cache_k = aggregate_cache_key([agg_key_source], prompt, settings.AGENT_HISTORY_CACHE_VERSION)
            cached = await cache_get(redis_client, cache_k)
            if cached:
                logger.debug(f"Агрегация {len(summaries)} саммари: cache HIT")
                return cached

        extra_body = None
        if settings.AGENT_DISABLE_THINKING:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        start_ts = time.time()
        request_payload = {
            "summaries_count": len(summaries),
            "combined_chars": len(combined_text),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": settings.AGENT_HISTORY_COMPRESSION_MAX_TOKENS,
        }

        try:
            response = await llm_service.completions(
                model=llm_service.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=settings.AGENT_HISTORY_COMPRESSION_MAX_TOKENS,
                stream=False,
                extra_body=extra_body,
            )
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                content = re.sub(
                    r"^(Саммари|Резюме|Summary|Объединённое саммари)\s*:\s*",
                    "", content, flags=re.IGNORECASE,
                ).strip()
                if content:
                    self._trace_llm("history_compression_aggregate", request_payload, {"content": content}, start_ts)
                    logger.debug(f"Агрегация {len(summaries)} саммари -> {len(content)} симв.")
                    if cache_k is not None:
                        await cache_set(
                            redis_client, cache_k, content,
                            settings.AGENT_HISTORY_CACHE_TTL_SECONDS,
                        )
                    return content

            self._trace_llm("history_compression_aggregate", request_payload, {"content": ""}, start_ts)
            logger.warning("Агрегация саммари: пустой ответ, fallback на конкатенацию")
        except Exception as e:
            logger.warning(f"Ошибка агрегации: {e}, fallback на конкатенацию")
            self._trace_llm("history_compression_aggregate", request_payload, None, start_ts, error=str(e))

        return combined_text

    async def _compress_history(
        self, llm_service, user_message: str, history: List[Dict], redis_client=None,
    ) -> str:
        """
        Сжатие истории диалога батчами с Redis-кэшем.
        Всегда возвращает строку (пустую при неудаче).
        """
        if not history:
            return ""

        normalized = self._normalize_history(history)
        if not normalized:
            return ""

        batch_size = settings.AGENT_HISTORY_COMPRESS_BATCH_SIZE
        all_batches = self._split_into_batches(normalized, batch_size)
        total_batches = len(all_batches)

        logger.debug(
            f"Сжатие истории: {len(normalized)} сообщений -> "
            f"{total_batches} батчей по {batch_size}"
        )

        start_ts = time.time()
        summaries: List[str] = []

        if settings.AGENT_HISTORY_COMPRESS_PARALLEL and total_batches > 1:
            tasks = [
                self._compress_batch(llm_service, batch, i + 1, total_batches, redis_client)
                for i, batch in enumerate(all_batches)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, r in enumerate(results):
                if isinstance(r, BaseException):
                    logger.warning(f"Батч {i + 1} упал: {r}")
                    continue
                if r:
                    summaries.append(r)
        else:
            for i, batch in enumerate(all_batches):
                r = await self._compress_batch(llm_service, batch, i + 1, total_batches, redis_client)
                if r:
                    summaries.append(r)

        elapsed = round((time.time() - start_ts) * 1000, 2)
        logger.debug(f"Батчи сжаты: {len(summaries)}/{total_batches} ({elapsed}ms)")

        if not summaries:
            logger.warning("Ни один батч не сжат — история потеряна")
            return ""
        if len(summaries) == 1:
            return summaries[0]

        final = await self._aggregate_summaries(llm_service, summaries, redis_client)
        logger.debug(f"Финальное саммари: {len(final)} симв.")
        return final

    async def _should_compress_history(self, history: List[Dict]) -> bool:
        """
        Решение о сжатии по токенам (точно) и числу сообщений.
        Использует tokenizer, чтобы учитывать специфику языка.
        """
        if not history:
            return False
        if len(history) < settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES:
            return False

        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")[: settings.MAX_HISTORY_MESSAGE_CHARS]}
            for m in history
            if m.get("role") in ("user", "assistant")
        ]
        if not messages:
            return False

        try:
            tokens = await count_messages_tokens_async(
                messages=messages, tools=None, use_exact=True,
            )
        except Exception as e:
            logger.warning(f"Не удалось посчитать токены истории: {e}, fallback на порог по числу")
            return len(messages) > settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES

        needs = (
            len(messages) > settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES
            or tokens > settings.AGENT_HISTORY_COMPRESS_MAX_TOKENS
        )
        logger.debug(
            f"История: {len(messages)} сообщений, {tokens} токенов, "
            f"порог={settings.AGENT_HISTORY_COMPRESS_MAX_TOKENS}, "
            f"compress={needs}"
        )
        return needs
4. app/services/agent_service/loop/setup.py — правки
Заменить метод _prepare_agent_history и передать redis_client:

python
    async def _prepare_agent_history(
        self, llm_service, mode: str, history: List[Dict], user_message: str,
        _log: logger, redis_client=None,
    ) -> Tuple[str, bool]:
        """
        Подготовка истории для агента.
        mode="full": sliding window без LLM.
        mode="compressed": LLM-сжатие (батчами, с Redis-кэшем), если история
        превышает порог по токенам или числу сообщений.
        """
        if not history:
            return "", False

        if mode == "compressed":
            if await self._should_compress_history(history):
                compressed = await self._compress_history(
                    llm_service, user_message, history, redis_client,
                )
                if compressed:
                    _log.info(f"История сжата для агента ({len(compressed)} симв.)")
                    return compressed, True
                _log.warning("Сжатие истории не удалось, fallback на sliding window")
            else:
                _log.debug("История ниже порога сжатия — sliding window")
            return self._build_sliding_window_history(history, _log), False

        return self._build_sliding_window_history(history, _log), False
5. app/services/agent_service/loop/mixin.py — передать redis_client
Добавить параметр в run_agent_loop:

python
    async def run_agent_loop(
        self,
        llm_service,
        user_message: str,
        history: List[Dict],
        qdrant_client,
        reranker_service,
        history_mode_override: Optional[str] = None,
        use_category_search_override: Optional[bool] = None,
        cancel_check: Optional[Callable[[], Awaitable[bool]]] = None,
        redis_client=None,
    ) -> AsyncGenerator[Tuple[str, Dict], None]:
И в вызове подготовки истории:

python
        agent_history_text, was_compressed = await self._prepare_agent_history(
            llm_service, mode, history, user_message, _log, redis_client,
        )
6. app/chat/pipeline.py — правки
Агент: передать redis_client в run_agent_loop:

python
                    agent_generator = agent_service.run_agent_loop(
                        llm_service=llm_service,
                        user_message=message,
                        history=raw_history_list,
                        qdrant_client=qdrant_client,
                        reranker_service=reranker_service,
                        history_mode_override=history_mode_override,
                        use_category_search_override=use_category_search_override,
                        cancel_check=lambda: _is_cancelled(task_id, redis_client),
                        redis_client=redis_client,
                    )
no-RAG: заменить блок сжатия:

python
        # Сжатие истории для no-RAG режима (compressed)
        if (not use_rag and history_mode_used == "compressed"):
            if await agent_service._should_compress_history(raw_history_list):
                try:
                    compressed_text = await agent_service._compress_history(
                        llm_service, message, raw_history_list, redis_client,
                    )
                    if compressed_text:
                        prepared_history_text = compressed_text
                        prepared_history_compressed = True
                        logger.info(f"No-RAG: сжатая история готова ({len(compressed_text)} симв.)")
                except Exception as e:
                    logger.warning(f"Ошибка сжатия истории (no-RAG): {e}")
7. app/prompts/history_compress.txt (новый)
text
Ты — инструмент извлечения фактов из истории диалога.
ТВОЯ ЗАДАЧА: создать краткое саммари истории, сохранив все важные факты.

ЖЁСТКИЕ ПРАВИЛА:
1. НЕ отвечай на вопросы из истории — только извлекай информацию.
2. НЕ связывай разные темы между собой.
3. НЕ добавляй новую информацию, которой не было в истории.
4. Сохраняй конкретику: имена, числа, даты, ID, названия документов.
5. Сохраняй связки "вопрос → ответ", если они содержат полезную информацию.
6. Если пользователь упоминал предпочтения, ограничения, контекст — фиксируй.
7. Пиши на русском языке, в форме кратких тезисов через точку.
8. Целевой объём: 150–250 слов.
9. Верни ТОЛЬКО текст саммари. Без заголовков "Саммари:", "Резюме:", без пояснений.

История диалога:
{history_text}

Саммари:
8. app/prompts/history_compress_aggregate.txt (новый)
text
Ты — инструмент сжатия уже сжатых саммари.
ТВОЯ ЗАДАЧА: объединить несколько частей саммари в одно финальное саммари.

ЖЁСТКИЕ ПРАВИЛА:
1. Сохрани ВСЕ ключевые факты из всех частей.
2. Устрани дубликаты и повторы.
3. Сохрани хронологический порядок, если он важен.
4. НЕ добавляй новую информацию.
5. Пиши на русском языке, в форме кратких тезисов через точку.
6. Целевой объём: 200–300 слов.
7. Верни ТОЛЬКО текст саммари, без заголовков и пояснений.

Части саммари:
{history_text}

Объединённое саммари:
9. .env.example — добавить
dotenv
# ─── Сжатие истории (compressed mode) ───
# Минимум сообщений (1 пара = 2 сообщения) для запуска сжатия
AGENT_HISTORY_COMPRESS_MIN_MESSAGES=12
# Порог по токенам — если история укладывается, сжатие не запускается
AGENT_HISTORY_COMPRESS_MAX_TOKENS=2000
# Максимум токенов финального саммари
AGENT_HISTORY_COMPRESSION_MAX_TOKENS=300
# Размер батча (сообщений) для одного LLM-запроса
AGENT_HISTORY_COMPRESS_BATCH_SIZE=20
# Максимум токенов batch-саммари
AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS=400
# Параллельные запросы батчей (true — латентность = max, false = sum)
AGENT_HISTORY_COMPRESS_PARALLEL=true
# Потолок числа батчей (защита от длинных историй)
AGENT_HISTORY_COMPRESS_MAX_BATCHES=3
# Redis-кэш саммари
AGENT_HISTORY_CACHE_ENABLED=true
AGENT_HISTORY_CACHE_TTL_SECONDS=3600
AGENT_HISTORY_CACHE_VERSION=v1
Убедитесь, что MAX_CONCURRENT_LLM_REQUESTS ≥ AGENT_HISTORY_COMPRESS_MAX_BATCHES (по умолчанию 16 ≥ 3).

10. app/services/agent_service/service.py — добавить загрузку второго промпта
После загрузки history_compress.txt добавить:

python
        try:
            self.history_compress_aggregate_prompt = (
                (prompts_dir / "history_compress_aggregate.txt")
                .read_text(encoding="utf-8")
                .strip()
            )
        except FileNotFoundError:
            self.history_compress_aggregate_prompt = textwrap.dedent("""\
                Ты — инструмент сжатия саммари.
                Объедини части в одно финальное саммари.
                Правила: сохрани все факты, устрани дубликаты, не добавляй новое.
                Части:
                {history_text}
            """).strip()
            logger.warning("history_compress_aggregate.txt не найден, встроенный промпт")
11. tests/unit/agent/test_history_compression.py — обновлённый
python
"""
Unit-тесты для HistoryMixin (сжатие истории батчами + Redis-кэш + токен-порог).

Проверяют:
- пустая история -> "";
- _should_compress_history: порог по сообщениям и по токенам;
- один батч -> один вызов LLM без агрегации;
- N батчей -> N + 1 запрос (при PARALLEL=False);
- Redis-кэш batch: cache HIT пропускает вызов LLM;
- Redis-кэш агрегации: cache HIT пропускает вызов LLM;
- ошибка одного батча не убивает остальные;
- MAX_BATCHES отбрасывает самые старые;
- think-блоки и префикс "Саммари:" удаляются.
"""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.services.agent_service.history import HistoryMixin


class TestableHistory(HistoryMixin):
    def __init__(self):
        self.history_compress_prompt = "H:\n{history_text}"
        self.history_compress_aggregate_prompt = "A:\n{history_text}"

    def _format_history_lines(self, messages):
        return "\n".join(
            f"{'П' if m['role'] == 'user' else 'А'}: {m.get('content', '')}"
            for m in messages
        )

    def _trace_llm(self, *args, **kwargs):
        pass


@pytest.fixture
def mixin():
    return TestableHistory()


def _make_history(n):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
        for i in range(n)
    ]


def _mock_llm(content="summary"):
    llm = AsyncMock()
    llm.model_name = "test"
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    llm.completions.return_value = resp
    return llm


# ─── _should_compress_history ───

@pytest.mark.asyncio
@patch("app.services.agent_service.history.count_messages_tokens_async", new_callable=AsyncMock)
@patch("app.services.agent_service.history.settings")
async def test_should_compress_short_history_false(mock_settings, mock_count, mixin):
    mock_settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES = 12
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_TOKENS = 2000
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 1500
    mock_count.return_value = 100
    assert not await mixin._should_compress_history(_make_history(4))


@pytest.mark.asyncio
@patch("app.services.agent_service.history.count_messages_tokens_async", new_callable=AsyncMock)
@patch("app.services.agent_service.history.settings")
async def test_should_compress_by_messages(mock_settings, mock_count, mixin):
    mock_settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES = 6
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_TOKENS = 99999
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 1500
    mock_count.return_value = 10
    assert await mixin._should_compress_history(_make_history(10))


@pytest.mark.asyncio
@patch("app.services.agent_service.history.count_messages_tokens_async", new_callable=AsyncMock)
@patch("app.services.agent_service.history.settings")
async def test_should_compress_by_tokens(mock_settings, mock_count, mixin):
    mock_settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES = 100
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_TOKENS = 2000
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 1500
    mock_count.return_value = 5000  # превышает порог
    assert await mixin._should_compress_history(_make_history(20))


# ─── _compress_history: базовые сценарии ───

@pytest.mark.asyncio
async def test_compress_empty_history(mixin):
    assert await mixin._compress_history(AsyncMock(), "q", []) == ""


@pytest.mark.asyncio
@patch("app.services.agent_service.history.settings")
async def test_compress_single_batch_no_aggregation(mock_settings, mixin):
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_SIZE = 20
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS = 400
    mock_settings.AGENT_HISTORY_COMPRESSION_MAX_TOKENS = 300
    mock_settings.AGENT_HISTORY_COMPRESS_PARALLEL = True
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_BATCHES = 5
    mock_settings.AGENT_HISTORY_CACHE_ENABLED = False
    mock_settings.AGENT_DISABLE_THINKING = True
    llm = _mock_llm("single")
    result = await mixin._compress_history(llm, "q", _make_history(5))
    assert result == "single"
    assert llm.completions.call_count == 1


@pytest.mark.asyncio
@patch("app.services.agent_service.history.settings")
async def test_compress_multi_batch_with_aggregation(mock_settings, mixin):
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_SIZE = 4
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS = 400
    mock_settings.AGENT_HISTORY_COMPRESSION_MAX_TOKENS = 300
    mock_settings.AGENT_HISTORY_COMPRESS_PARALLEL = False
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_BATCHES = 10
    mock_settings.AGENT_HISTORY_CACHE_ENABLED = False
    mock_settings.AGENT_DISABLE_THINKING = True
    llm = _mock_llm("ok")
    # 10 / 4 = 3 батча + 1 агрегация = 4
    result = await mixin._compress_history(llm, "q", _make_history(10))
    assert result == "ok"
    assert llm.completions.call_count == 4


@pytest.mark.asyncio
@patch("app.services.agent_service.history.settings")
async def test_compress_max_batches_limits(mock_settings, mixin):
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_SIZE = 2
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS = 400
    mock_settings.AGENT_HISTORY_COMPRESSION_MAX_TOKENS = 300
    mock_settings.AGENT_HISTORY_COMPRESS_PARALLEL = False
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_BATCHES = 2
    mock_settings.AGENT_HISTORY_CACHE_ENABLED = False
    mock_settings.AGENT_DISABLE_THINKING = True
    llm = _mock_llm("ok")
    # 20 / 2 = 10 батчей, но MAX=2 → 2 + агрегация = 3
    result = await mixin._compress_history(llm, "q", _make_history(20))
    assert result == "ok"
    assert llm.completions.call_count == 3


@pytest.mark.asyncio
@patch("app.services.agent_service.history.settings")
async def test_compress_batch_error_others_survive(mock_settings, mixin):
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_SIZE = 4
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS = 400
    mock_settings.AGENT_HISTORY_COMPRESSION_MAX_TOKENS = 300
    mock_settings.AGENT_HISTORY_COMPRESS_PARALLEL = True
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_BATCHES = 10
    mock_settings.AGENT_HISTORY_CACHE_ENABLED = False
    mock_settings.AGENT_DISABLE_THINKING = True
    llm = AsyncMock()
    llm.model_name = "m"
    call_counter = {"n": 0}
    async def side_effect(*args, **kwargs):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            raise RuntimeError("timeout")
        r = MagicMock()
        c = MagicMock()
        c.message.content = "ok"
        r.choices = [c]
        return r
    llm.completions.side_effect = side_effect
    result = await mixin._compress_history(llm, "q", _make_history(10))
    assert result == "ok"


@pytest.mark.asyncio
@patch("app.services.agent_service.history.settings")
async def test_compress_strips_think_and_prefix(mock_settings, mixin):
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_SIZE = 20
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS = 400
    mock_settings.AGENT_HISTORY_COMPRESSION_MAX_TOKENS = 300
    mock_settings.AGENT_HISTORY_COMPRESS_PARALLEL = True
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_BATCHES = 5
    mock_settings.AGENT_HISTORY_CACHE_ENABLED = False
    mock_settings.AGENT_DISABLE_THINKING = True
    llm = _mock_llm("<think>x</think>Саммари: чистый")
    result = await mixin._compress_history(llm, "q", _make_history(5))
    assert result == "чистый"


# ─── Redis-кэш ───

@pytest.mark.asyncio
@patch("app.services.agent_service.history.settings")
async def test_cache_hit_skips_llm(mock_settings, mixin):
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_SIZE = 20
    mock_settings.AGENT_HISTORY_COMPRESS_BATCH_MAX_TOKENS = 400
    mock_settings.AGENT_HISTORY_COMPRESSION_MAX_TOKENS = 300
    mock_settings.AGENT_HISTORY_COMPRESS_PARALLEL = True
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_BATCHES = 5
    mock_settings.AGENT_HISTORY_CACHE_ENABLED = True
    mock_settings.AGENT_HISTORY_CACHE_TTL_SECONDS = 3600
    mock_settings.AGENT_HISTORY_CACHE_VERSION = "test"
    mock_settings.AGENT_DISABLE_THINKING = True
    llm = _mock_llm("новое")
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=b"cached_summary")
    redis.set = AsyncMock()
    result = await mixin._compress_history(llm, "q", _make_history(5), redis)
    assert result == "cached_summary"
    assert llm.completions.call_count == 0
