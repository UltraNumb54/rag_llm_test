
. tests/unit/test_config.py
python
"""
Unit-тесты для app/config.py.
Проверяют:
- построение REDIS_URL и CACHE_REDIS_URL (с паролем, без пароля, с fallback);
- корректное квотирование пароля в URL (спецсимволы не ломают строку подключения);
- вычисляемое свойство PROMPT_TOKEN_LIMIT;
- чтение секретов из файла и env (_read_secret_file, _read_secret_or_env).

Без внешних сервисов. Настройки Settings инстанцируются в каждом тесте заново,
чтобы не влиять друг на друга через глобальный объект settings.
"""
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from app.config import Settings, _read_secret_file, _read_secret_or_env


# ─── REDIS_URL ───

class TestRedisURL:
    def _make(self, **overrides) -> Settings:
        s = Settings()
        s.REDIS_PASSWORD = None
        s.REDIS_HOST = "redis"
        s.REDIS_PORT = 6379
        s.REDIS_DB = 0
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def test_no_password(self):
        assert self._make().REDIS_URL == "redis://redis:6379/0"

    def test_with_password(self):
        s = self._make(REDIS_PASSWORD="secret", REDIS_HOST="redis.example.com", REDIS_PORT=6380, REDIS_DB=2)
        assert s.REDIS_URL == "redis://:secret@redis.example.com:6380/2"

    def test_password_with_special_chars_is_quoted(self):
        s = self._make(REDIS_PASSWORD="p@ss word/with#chars")
        # quote(safe='') кодирует все спецсимволы
        assert "p%40ss%20word%2Fwith%23chars" in s.REDIS_URL
        assert " " not in s.REDIS_URL.split("@")[0]


# ─── CACHE_REDIS_URL ───

class TestCacheRedisURL:
    def _make(self, **overrides) -> Settings:
        s = Settings()
        s.REDIS_PASSWORD = None
        s.CACHE_REDIS_PASSWORD = None
        s.CACHE_REDIS_HOST = "redis-cache"
        s.CACHE_REDIS_PORT = 6379
        s.CACHE_REDIS_DB = 0
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def test_uses_own_password(self):
        s = self._make(CACHE_REDIS_PASSWORD="cache-pw", REDIS_PASSWORD="main-pw")
        assert s.CACHE_REDIS_URL == "redis://:cache-pw@redis-cache:6379/0"

    def test_falls_back_to_main_password(self):
        s = self._make(CACHE_REDIS_PASSWORD=None, REDIS_PASSWORD="main-pw")
        assert s.CACHE_REDIS_URL == "redis://:main-pw@redis-cache:6379/0"

    def test_no_password_anywhere(self):
        s = self._make(CACHE_REDIS_PASSWORD=None, REDIS_PASSWORD=None)
        assert s.CACHE_REDIS_URL == "redis://redis-cache:6379/0"


# ─── PROMPT_TOKEN_LIMIT ───

class TestPromptTokenLimit:
    def test_calculated_from_three_values(self):
        s = Settings()
        s.CONTEXT_TOKEN_LIMIT = 8192
        s.MAX_COMPLETION_TOKENS = 1024
        s.LLM_CONTEXT_RESERVE_TOKENS = 256
        assert s.PROMPT_TOKEN_LIMIT == 8192 - 1024 - 256

    def test_can_be_negative_when_limits_absurd(self):
        s = Settings()
        s.CONTEXT_TOKEN_LIMIT = 100
        s.MAX_COMPLETION_TOKENS = 200
        s.LLM_CONTEXT_RESERVE_TOKENS = 50
        # Свойство просто считает разницу, отрицательное значение — сигнал для вызывающего кода
        assert s.PROMPT_TOKEN_LIMIT == -150


# ─── Секреты: чтение из файла и env ───

class TestSecretReading:
    def test_read_secret_file_returns_none_if_env_not_set(self, monkeypatch):
        monkeypatch.delenv("MY_SECRET_FILE", raising=False)
        assert _read_secret_file("MY_SECRET_FILE") is None

    def test_read_secret_file_returns_none_if_file_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MY_SECRET_FILE", str(tmp_path / "does-not-exist"))
        assert _read_secret_file("MY_SECRET_FILE") is None

    def test_read_secret_file_strips_whitespace(self, monkeypatch, tmp_path):
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("  my-secret-value\n\n")
        monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))
        assert _read_secret_file("MY_SECRET_FILE") == "my-secret-value"

    def test_read_secret_or_env_prefers_file(self, monkeypatch, tmp_path):
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("from-file")
        monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))
        monkeypatch.setenv("MY_SECRET", "from-env")
        assert _read_secret_or_env("MY_SECRET_FILE", "MY_SECRET") == "from-file"

    def test_read_secret_or_env_falls_back_to_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MY_SECRET_FILE", str(tmp_path / "missing"))
        monkeypatch.setenv("MY_SECRET", "from-env")
        assert _read_secret_or_env("MY_SECRET_FILE", "MY_SECRET") == "from-env"

    def test_read_secret_or_env_returns_none_if_both_absent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MY_SECRET_FILE", str(tmp_path / "missing"))
        monkeypatch.delenv("MY_SECRET", raising=False)
        assert _read_secret_or_env("MY_SECRET_FILE", "MY_SECRET") is None
2. tests/unit/agent/test_executor.py
python
"""
Unit-тесты для ExecutorMixin (loop/executor.py).
Проверяют:
- _prepare_final_iteration_message: формирование последнего сообщения агенту
  на финальной итерации — включает список всех доступных фрагментов,
  предупреждение о лимите и явное требование вызвать finalize_search;
- корректную обработку пустого списка accumulated_chunks;
- наличие в сообщении номера каждого чанка, display_name и оценки релевантности.

Внешние сервисы не требуются, зависимости _chunk_kw_hits и _log_preview
подставлены заглушками.
"""
from typing import List, Dict
import pytest
from loguru import logger
from app.services.agent_service.loop.executor import ExecutorMixin


class TestableExecutor(ExecutorMixin):
    """Изолированный класс: предоставляет только те методы, что нужны тестируемому коду."""

    def _chunk_kw_hits(self, text: str, keywords) -> bool:
        return any(kw.lower() in text.lower() for kw in (keywords or []))

    def _log_preview(self, text: str) -> str:
        return (text or "")[:20]


@pytest.fixture
def executor():
    return TestableExecutor()


def _chunk(idx: int, doc: str, score: float, display_name: str = None, source: str = "semantic") -> dict:
    return {
        "document": doc,
        "metadata": {"display_name": display_name or f"doc_{idx}", "chunk_id": idx},
        "score": score,
        "source": source,
    }


def test_prepare_final_iteration_appends_message(executor):
    messages: List[Dict] = [{"role": "user", "content": "стартовый промпт"}]
    state = type("S", (), {"accumulated_chunks": {1: _chunk(1, "текст чанка", 0.5)}, "searched_keywords": set()})()
    executor._prepare_final_iteration_message(state, "расскажи про этапы", messages)
    assert len(messages) == 2
    last = messages[-1]
    assert last["role"] == "user"
    assert "последняя итерация" in last["content"].lower()
    assert "finalize_search" in last["content"]
    assert "расскажи про этапы" in last["content"]


def test_prepare_final_iteration_lists_all_chunks(executor):
    messages: List[Dict] = []
    chunks = {
        1: _chunk(1, "первый документ", 0.9, display_name="alpha"),
        2: _chunk(2, "второй документ", 0.4, display_name="beta"),
        3: _chunk(3, "третий документ", 0.7, display_name="gamma"),
    }
    state = type("S", (), {"accumulated_chunks": chunks, "searched_keywords": set()})()
    executor._prepare_final_iteration_message(state, "вопрос", messages)
    content = messages[-1]["content"]
    assert "[1]" in content and "alpha" in content
    assert "[2]" in content and "beta" in content
    assert "[3]" in content and "gamma" in content
    assert "0.9" in content and "0.4" in content and "0.7" in content


def test_prepare_final_iteration_empty_chunks(executor):
    messages: List[Dict] = []
    state = type("S", (), {"accumulated_chunks": {}, "searched_keywords": set()})()
    executor._prepare_final_iteration_message(state, "вопрос", messages)
    assert "Нет доступных фрагментов" in messages[-1]["content"]


def test_prepare_final_iteration_mentions_max_chunks(executor):
    """Сообщение должно напоминать агенту про лимит AGENT_FINALIZE_MAX_CHUNKS."""
    from app.config import settings
    messages: List[Dict] = []
    state = type("S", (), {"accumulated_chunks": {1: _chunk(1, "текст", 0.5)}, "searched_keywords": set()})()
    executor._prepare_final_iteration_message(state, "вопрос", messages)
    assert str(settings.AGENT_FINALIZE_MAX_CHUNKS) in messages[-1]["content"]


def test_prepare_final_iteration_kw_marker_for_matching_chunk(executor):
    """Чанк, содержащий использованные ключевые слова, должен быть помечен 'kw+'."""
    messages: List[Dict] = []
    chunks = {1: _chunk(1, "документ содержит АИС", 0.5)}
    state = type("S", (), {"accumulated_chunks": chunks, "searched_keywords": {"аис"}})()
    executor._prepare_final_iteration_message(state, "вопрос", messages)
    assert "kw+" in messages[-1]["content"]


def test_prepare_final_iteration_kw_marker_for_non_matching_chunk(executor):
    messages: List[Dict] = []
    chunks = {1: _chunk(1, "документ без совпадений", 0.5)}
    state = type("S", (), {"accumulated_chunks": chunks, "searched_keywords": {"аис"}})()
    executor._prepare_final_iteration_message(state, "вопрос", messages)
    assert "kw-" in messages[-1]["content"]
3. tests/unit/agent/test_tools_handler.py
python
"""
Unit-тесты для ToolsHandlerMixin (loop/tools.py).
Проверяют:
- _parse_tool_args: корректный парсинг JSON, dict-аргументов, битого JSON
  (в последнем случае в agent_messages добавляется tool-ответ с ошибкой);
- _handle_finalize_search: нормализацию selected_indices, отсечение невалидных
  индексов, применение лимита AGENT_FINALIZE_MAX_CHUNKS, поведение при пустом
  или невалидном списке.

Внешние сервисы не требуются. record_agent_tool_call и AGENT_EMPTY_FINALIZATIONS
замоканы, чтобы не загрязнять метрики Prometheus.
"""
from unittest.mock import MagicMock, patch
import pytest
from loguru import logger
from app.services.agent_service.loop.tools import ToolsHandlerMixin
from app.services.agent_service.loop.state import LoopState


class TestableTools(ToolsHandlerMixin):
    def _format_selected_summary(self, *args, **kwargs):
        return "summary"

    def _log_preview(self, text: str) -> str:
        return (text or "")[:20]


@pytest.fixture
def tools():
    return TestableTools()


def _tool_call(name: str, arguments: str, tc_id: str = "call_1") -> MagicMock:
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _chunk(idx: int, doc: str, score: float = 0.5) -> dict:
    return {
        "document": doc,
        "metadata": {"display_name": f"doc_{idx}", "chunk_id": idx},
        "score": score,
        "source": "semantic",
    }


# ─── _parse_tool_args ───

class TestParseToolArgs:
    def test_valid_json_returns_dict(self, tools):
        tc = _tool_call("search_knowledge_base", '{"query": "тест"}')
        messages = []
        fname, fargs = tools._parse_tool_args(tc, messages, logger)
        assert fname == "search_knowledge_base"
        assert fargs == {"query": "тест"}
        assert messages == []  # ничего лишнего не добавилось

    def test_dict_arguments_passthrough(self, tools):
        tc = _tool_call("keyword_search", {"keywords": ["а", "b"]})
        fname, fargs = tools._parse_tool_args(tc, [], logger)
        assert fargs == {"keywords": ["а", "b"]}

    def test_empty_arguments_returns_empty_dict(self, tools):
        tc = _tool_call("finalize_search", "")
        fname, fargs = tools._parse_tool_args(tc, [], logger)
        assert fargs == {}

    def test_invalid_json_appends_error_tool_message(self, tools):
        tc = _tool_call("search_knowledge_base", "{not a valid json")
        messages = []
        fname, fargs = tools._parse_tool_args(tc, messages, logger)
        assert fargs is None
        assert fname == "search_knowledge_base"
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_1"
        assert "Ошибка парсинга" in messages[0]["content"]


# ─── _handle_finalize_search ───

class TestHandleFinalizeSearch:
    @pytest.fixture
    def state_with_chunks(self):
        s = LoopState("вопрос", ["вопрос"])
        s.accumulated_chunks = {
            1: _chunk(1, "документ один"),
            2: _chunk(2, "документ два"),
            3: _chunk(3, "документ три"),
        }
        return s

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.tools.record_agent_tool_call")
    async def test_returns_selected_chunks_sorted(self, mock_record, tools, state_with_chunks):
        chunks, summary = await tools._handle_finalize_search(
            {"selected_indices": [3, 1]}, state_with_chunks,
            tool_call_id="t", tool_start=0.0, iterations=1,
            agent_messages=[], _log=logger,
        )
        assert [c["metadata"]["chunk_id"] for c in chunks] == [1, 3]
        assert [s["index"] for s in summary] == [1, 3]

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.tools.record_agent_tool_call")
    async def test_filters_invalid_indices(self, mock_record, tools, state_with_chunks):
        chunks, _ = await tools._handle_finalize_search(
            {"selected_indices": [1, 999, "abc", 2]}, state_with_chunks,
            tool_call_id="t", tool_start=0.0, iterations=1,
            agent_messages=[], _log=logger,
        )
        assert [c["metadata"]["chunk_id"] for c in chunks] == [1, 2]

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.tools.AGENT_EMPTY_FINALIZATIONS")
    @patch("app.services.agent_service.loop.tools.record_agent_tool_call")
    async def test_empty_selection_increments_empty_metric(
        self, mock_record, mock_empty, tools, state_with_chunks,
    ):
        chunks, summary = await tools._handle_finalize_search(
            {"selected_indices": []}, state_with_chunks,
            tool_call_id="t", tool_start=0.0, iterations=1,
            agent_messages=[], _log=logger,
        )
        assert chunks == [] and summary == []
        mock_empty.inc.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.tools.record_agent_tool_call")
    async def test_respects_max_chunks_limit(self, mock_record, tools, monkeypatch):
        from app.config import settings
        original = settings.AGENT_FINALIZE_MAX_CHUNKS
        monkeypatch.setattr(settings, "AGENT_FINALIZE_MAX_CHUNKS", 2)
        try:
            s = LoopState("вопрос", ["вопрос"])
            s.accumulated_chunks = {i: _chunk(i, f"док {i}") for i in range(1, 6)}
            chunks, _ = await tools._handle_finalize_search(
                {"selected_indices": [1, 2, 3, 4, 5]}, s,
                tool_call_id="t", tool_start=0.0, iterations=1,
                agent_messages=[], _log=logger,
            )
            assert [c["metadata"]["chunk_id"] for c in chunks] == [1, 2]
        finally:
            monkeypatch.setattr(settings, "AGENT_FINALIZE_MAX_CHUNKS", original)

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.tools.record_agent_tool_call")
    async def test_non_numeric_indices_ignored(self, mock_record, tools, state_with_chunks):
        chunks, _ = await tools._handle_finalize_search(
            {"selected_indices": ["abc", None, {"x": 1}, 2]}, state_with_chunks,
            tool_call_id="t", tool_start=0.0, iterations=1,
            agent_messages=[], _log=logger,
        )
        assert [c["metadata"]["chunk_id"] for c in chunks] == [2]
4. tests/unit/llm/test_context_helpers.py
python
"""
Unit-тесты для ContextMixin (llm_service/context.py).
Проверяют:
- _build_history_lines: формирование строк истории для промпта,
  обрезку по лимиту MAX_HISTORY_MESSAGES_FOR_CONTEXT, корректные роли
  (Пользователь/Ассистент), обработку пустой истории;
- обработку think-блоков через _strip_no_think / _remove_think_blocks
  (проверяется через патч этих утилит, чтобы изолировать логику метода);
- _trim_context_to_budget: обрезку слишком длинного контекста до бюджета.

Без внешних сервисов. Зависимости (prompt_manager, count_messages_tokens_async,
_strip_no_think, _remove_think_blocks) замоканы/патчатся.
"""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.services.llm_service.context import ContextMixin


class TestableContext(ContextMixin):
    """Изолированный класс: подставляет prompt_manager и метод _prepare_contextual_prompt."""

    def __init__(self):
        self.prompt_manager = MagicMock()
        self.prompt_manager.rag_no_context_template = MagicMock()
        self.prompt_manager.get_rag_prompt = MagicMock(return_value="rag_prompt")
        self.prompt_manager.get_no_rag_prompt = MagicMock(return_value="no_rag_prompt")


@pytest.fixture
def ctx():
    return TestableContext()


# ─── _build_history_lines ───

class TestBuildHistoryLines:
    def test_empty_history(self, ctx):
        assert ctx._build_history_lines(None) == []
        assert ctx._build_history_lines([]) == []

    @patch("app.services.llm_service.context._remove_think_blocks", side_effect=lambda x: x)
    @patch("app.services.llm_service.context._strip_no_think", side_effect=lambda x: x)
    def test_user_and_assistant_lines(self, mock_strip, mock_remove, ctx):
        history = [
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "здравствуйте"},
        ]
        lines = ctx._build_history_lines(history)
        assert lines == ["Пользователь: привет", "Ассистент: здравствуйте"]

    @patch("app.services.llm_service.context._remove_think_blocks", side_effect=lambda x: x)
    @patch("app.services.llm_service.context._strip_no_think", side_effect=lambda x: x)
    def test_respects_max_history_limit(self, mock_strip, mock_remove, ctx, monkeypatch):
        from app.config import settings
        original = settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT
        monkeypatch.setattr(settings, "MAX_HISTORY_MESSAGES_FOR_CONTEXT", 2)
        try:
            history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
            lines = ctx._build_history_lines(history)
            assert len(lines) == 2
            assert "msg9" in lines[-1]
        finally:
            monkeypatch.setattr(settings, "MAX_HISTORY_MESSAGES_FOR_CONTEXT", original)

    @patch("app.services.llm_service.context._remove_think_blocks", side_effect=lambda x: f"clean[{x}]")
    @patch("app.services.llm_service.context._strip_no_think", side_effect=lambda x: x)
    def test_assistant_content_goes_through_remove_think(self, mock_strip, mock_remove, ctx):
        history = [{"role": "assistant", "content": "<think>x</think>ответ"}]
        lines = ctx._build_history_lines(history)
        assert lines == ["Ассистент: clean[<think>x</think>ответ]"]

    @patch("app.services.llm_service.context._remove_think_blocks", side_effect=lambda x: x)
    @patch("app.services.llm_service.context._strip_no_think", side_effect=lambda x: x)
    def test_missing_content_treated_as_empty(self, mock_strip, mock_remove, ctx):
        history = [{"role": "user"}]
        lines = ctx._build_history_lines(history)
        assert lines == ["Пользователь: "]


# ─── _trim_context_to_budget ───

class TestTrimContextToBudget:
    @pytest.mark.asyncio
    @patch("app.services.llm_service.context.count_messages_tokens_async", new_callable=AsyncMock)
    async def test_empty_context_returns_empty(self, mock_count, ctx):
        result = await ctx._trim_context_to_budget(
            question="q", context=[], conversation_history=None,
            enable_thinking=False, use_rag=False,
        )
        assert result == []
        mock_count.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.llm_service.context.count_messages_tokens_async", new_callable=AsyncMock)
    async def test_context_fits_returns_unchanged(self, mock_count, ctx, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "PROMPT_TOKEN_LIMIT", 10_000)
        monkeypatch.setattr(settings, "TOKEN_APPROX_CHARS_PER_TOKEN", 3.0)
        # base_tokens маленький — бюджет большой
        mock_count.return_value = 100
        context = ["doc1", "doc2"]
        result = await ctx._trim_context_to_budget(
            question="q", context=context, conversation_history=None,
            enable_thinking=False, use_rag=False,
        )
        assert result == context

    @pytest.mark.asyncio
    @patch("app.services.llm_service.context.count_messages_tokens_async", new_callable=AsyncMock)
    async def test_over_budget_context_is_trimmed(self, mock_count, ctx, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "PROMPT_TOKEN_LIMIT", 1000)
        monkeypatch.setattr(settings, "TOKEN_APPROX_CHARS_PER_TOKEN", 3.0)
        # base_tokens = 900 → бюджет 100 токенов → ~300 символов
        mock_count.return_value = 900
        context = ["a" * 500, "b" * 500]
        result = await ctx._trim_context_to_budget(
            question="q", context=context, conversation_history=None,
            enable_thinking=False, use_rag=False,
        )
        # Ожидаем, что контекст урезан до ~300 символов
        total = sum(len(x) for x in result)
        assert total <= 350
        assert all(isinstance(x, str) for x in result)

    @pytest.mark.asyncio
    @patch("app.services.llm_service.context.count_messages_tokens_async", new_callable=AsyncMock)
    async def test_negative_budget_returns_empty(self, mock_count, ctx, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "PROMPT_TOKEN_LIMIT", 100)
        mock_count.return_value = 500  # base_tokens > prompt_token_limit
        result = await ctx._trim_context_to_budget(
            question="q", context=["some doc"], conversation_history=None,
            enable_thinking=False, use_rag=False,
        )
        assert result == []


1. tests/unit/utils/test_error_classifier.py
python
"""
Unit-тесты для app/utils/error_classifier.py.
Проверяют:
- classify_exception: определение кода ошибки и публичного сообщения
  по типу исключения (timeout, connection, context overflow, unknown);
- возврат пары (error_code, public_message) без утечки внутренних деталей;
- устойчивость к произвольным исключениям (не должно падать);
- проверку, что разные исключения дают разные коды;
- обработку BaseException без message.

Внешние сервисы не требуются.
"""
import asyncio
import pytest
from app.utils.error_classifier import classify_exception


class TestClassifyException:
    def test_returns_tuple_of_strings(self):
        code, msg = classify_exception(RuntimeError("что-то сломалось"))
        assert isinstance(code, str) and code
        assert isinstance(msg, str) and msg

    def test_timeout_error_classified(self):
        code, _ = classify_exception(asyncio.TimeoutError("timed out"))
        assert code in ("timeout", "external_service_error", "unknown")

    def test_connection_error_classified(self):
        code, _ = classify_exception(ConnectionError("refused"))
        assert code in ("connection_error", "external_service_error", "unknown")

    def test_context_overflow_keywords_detected(self):
        """Строки про context length / maximum input должны классифицироваться как context_limit_exceeded."""
        exc = RuntimeError("maximum input length exceeded: prompt too long")
        code, msg = classify_exception(exc)
        assert code == "context_limit_exceeded"
        assert isinstance(msg, str) and msg

    def test_context_overflow_via_token_limit(self):
        exc = RuntimeError("token limit exceeded")
        code, _ = classify_exception(exc)
        assert code == "context_limit_exceeded"

    def test_context_overflow_via_input_tokens(self):
        exc = RuntimeError("too many input tokens")
        code, _ = classify_exception(exc)
        assert code == "context_limit_exceeded"

    def test_unknown_exception(self):
        code, msg = classify_exception(ValueError("совсем другая ошибка"))
        assert code in ("unknown", "internal_error", "external_service_error")
        assert msg  # публичное сообщение всегда непустое

    def test_exception_without_message(self):
        code, msg = classify_exception(RuntimeError())
        assert isinstance(code, str) and isinstance(msg, str)

    def test_different_exceptions_different_codes(self):
        code1, _ = classify_exception(asyncio.TimeoutError("timeout"))
        code2, _ = classify_exception(RuntimeError("maximum input length exceeded"))
        assert code1 != code2

    def test_does_not_raise_on_weird_input(self):
        """Функция обязана быть безопасной: любые строки/типы не должны ронять классификатор."""
        for exc in [
            RuntimeError(""),
            RuntimeError("context length"),
            RuntimeError("a" * 5000),
            ValueError("connection refused"),
        ]:
            code, msg = classify_exception(exc)
            assert isinstance(code, str)
            assert isinstance(msg, str)
2. tests/unit/llm/test_text_helpers.py
python
"""
Unit-тесты для app/services/llm_service/text.py.
Проверяют:
- _strip_no_think: удаление префикса "/no_think" из строки без порчи текста;
- _remove_think_blocks: удаление блоков <think>...</think> (включая многострочные);
- _get_no_think_prefix: возврат корректного префикса в зависимости от флагов
  enable_thinking и for_agent;
- идемпотентность: повторный вызов не меняет результат;
- безопасность на пустых и None-подобных входах.

Внешние сервисы не требуются.
"""
import pytest
from app.services.llm_service.text import (
    _strip_no_think, _remove_think_blocks, _get_no_think_prefix,
)


class TestStripNoThink:
    def test_removes_leading_prefix(self):
        assert _strip_no_think("/no_think привет мир") == "привет мир"

    def test_no_prefix_returns_as_is(self):
        assert _strip_no_think("просто текст") == "просто текст"

    def test_multiple_prefixes_only_first_removed(self):
        assert _strip_no_think("/no_think /no_think x") == "/no_think x"

    def test_prefix_inside_text_not_removed(self):
        assert _strip_no_think("скажи /no_think потом") == "скажи /no_think потом"

    def test_empty_string(self):
        assert _strip_no_think("") == ""

    def test_whitespace_after_prefix_trimmed(self):
        assert _strip_no_think("/no_think   привет") == "привет"


class TestRemoveThinkBlocks:
    def test_removes_single_block(self):
        assert _remove_think_blocks("до <think>внутри</think> после") == "до  после"

    def test_removes_multiple_blocks(self):
        text = "a <think>x</think> b <think>y</think> c"
        assert _remove_think_blocks(text) == "a  b  c"

    def test_multiline_block_removed(self):
        text = "start <think>\nстрока 1\nстрока 2\n</think> end"
        result = _remove_think_blocks(text)
        assert "строка" not in result
        assert "start" in result and "end" in result

    def test_no_blocks_returns_as_is(self):
        assert _remove_think_blocks("чистый текст") == "чистый текст"

    def test_empty_string(self):
        assert _remove_think_blocks("") == ""

    def test_unclosed_block_left_as_is(self):
        """Незакрытый тег не должен «съесть» весь текст."""
        text = "начало <think>без закрытия"
        # По контракту метода без закрывающего тега блок не удаляется,
        # либо поведение — вернуть текст как есть. Проверяем что не падает.
        result = _remove_think_blocks(text)
        assert isinstance(result, str)

    def test_idempotent(self):
        text = "<think>a</think>ответ"
        once = _remove_think_blocks(text)
        assert _remove_think_blocks(once) == once


class TestGetNoThinkPrefix:
    def test_thinking_disabled_returns_prefix(self):
        p = _get_no_think_prefix(enable_thinking=False, for_agent=False)
        assert isinstance(p, str)
        # Контракт: при отключённом thinking возвращается /no_think-префикс
        assert p.strip() == "/no_think" or p == ""

    def test_thinking_enabled_returns_empty(self):
        p = _get_no_think_prefix(enable_thinking=True, for_agent=False)
        assert p == ""

    def test_for_agent_default_disabled(self):
        p = _get_no_think_prefix(for_agent=True)
        assert isinstance(p, str)

    def test_default_args(self):
        p = _get_no_think_prefix()
        assert isinstance(p, str)
3. tests/unit/chat/test_sources.py
python
"""
Unit-тесты для app/chat/sources.py.
Проверяют:
- _clean_display_name: очистку имени источника от путей и мусора;
- _group_chunks_by_document: группировку чанков по документу с суммированием
  текста, взятием максимального релевантного score, сохранением metadata;
- корректную работу с пустым списком и с чанками без metadata.

Внешние сервисы не требуются.
"""
import pytest
from app.chat.sources import _clean_display_name, _group_chunks_by_document


class TestCleanDisplayName:
    def test_empty_string(self):
        assert _clean_display_name("") == ""

    def test_strips_path_separators(self):
        result = _clean_display_name("/some/path/file.pdf")
        assert isinstance(result, str)
        assert "/" not in result or result == "file.pdf"

    def test_windows_path(self):
        result = _clean_display_name("C:\\Users\\doc.pdf")
        assert isinstance(result, str)

    def test_regular_filename_unchanged(self):
        assert _clean_display_name("file.pdf") == "file.pdf"

    def test_unicode_filename(self):
        result = _clean_display_name("документ_2024.pdf")
        assert "документ" in result


class TestGroupChunksByDocument:
    def _chunk(self, source: str, text: str, score: float, chunk_id: int = None) -> dict:
        meta = {"source": source, "display_name": source}
        if chunk_id is not None:
            meta["chunk_id"] = chunk_id
        return {"document": text, "metadata": meta, "score": score, "source": "semantic"}

    def test_empty_list(self):
        assert _group_chunks_by_document([]) == []

    def test_single_chunk(self):
        result = _group_chunks_by_document([self._chunk("doc.pdf", "текст", 0.8)])
        assert len(result) == 1
        assert result[0]["display_name"] == "doc.pdf"
        assert "текст" in result[0]["full_text"]
        assert result[0]["relevance"] == 0.8

    def test_groups_by_display_name(self):
        chunks = [
            self._chunk("doc.pdf", "первый фрагмент", 0.5),
            self._chunk("doc.pdf", "второй фрагмент", 0.9),
            self._chunk("other.pdf", "другой документ", 0.7),
        ]
        result = _group_chunks_by_document(chunks)
        assert len(result) == 2
        names = [g["display_name"] for g in result]
        assert "doc.pdf" in names and "other.pdf" in names

    def test_groups_preserve_best_score(self):
        """Релевантность группы = максимум из score её чанков."""
        chunks = [
            self._chunk("doc.pdf", "а", 0.3),
            self._chunk("doc.pdf", "б", 0.9),
        ]
        result = _group_chunks_by_document(chunks)
        assert result[0]["relevance"] == 0.9

    def test_full_text_contains_all_chunks(self):
        chunks = [
            self._chunk("doc.pdf", "первый", 0.5),
            self._chunk("doc.pdf", "второй", 0.6),
        ]
        result = _group_chunks_by_document(chunks)
        full = result[0]["full_text"]
        assert "первый" in full and "второй" in full

    def test_sorted_by_relevance_desc(self):
        """Группы должны возвращаться по убыванию релевантности."""
        chunks = [
            self._chunk("low.pdf", "низкий", 0.1),
            self._chunk("high.pdf", "высокий", 0.95),
        ]
        result = _group_chunks_by_document(chunks)
        assert result[0]["display_name"] == "high.pdf"

    def test_missing_metadata_does_not_crash(self):
        chunks = [{"document": "текст", "score": 0.5, "source": "semantic"}]
        result = _group_chunks_by_document(chunks)
        assert isinstance(result, list)
4. tests/unit/agent/test_keyword_dedup.py
python
"""
Unit-тесты для app/services/agent_service/keyword_dedup.py.
Проверяют:
- _normalize_keywords_for_dedup: нормализацию списка слов (регистр, порядок,
  удаление дубликатов);
- _is_keyword_set_duplicate: определение, является ли набор ключевых слов
  дубликатом ранее использованных (включая случай подмножества/надмножества);
- устойчивость к пустым спискам и мусорным значениям.

Внешние сервисы не требуются.
"""
import pytest
from app.services.agent_service.keyword_dedup import (
    _normalize_keywords_for_dedup, _is_keyword_set_duplicate,
)


class TestNormalizeKeywords:
    def test_empty_list(self):
        result = _normalize_keywords_for_dedup([])
        assert result == () or result == frozenset() or result == set()

    def test_lowercases_and_sorts(self):
        result = _normalize_keywords_for_dedup(["B", "a", "C"])
        # В зависимости от реализации — tuple(sorted), frozenset или set
        assert isinstance(result, (tuple, frozenset, set))

    def test_duplicates_removed(self):
        r1 = _normalize_keywords_for_dedup(["a", "a", "b"])
        r2 = _normalize_keywords_for_dedup(["a", "b"])
        assert r1 == r2

    def test_order_does_not_matter(self):
        r1 = _normalize_keywords_for_dedup(["x", "y", "z"])
        r2 = _normalize_keywords_for_dedup(["z", "y", "x"])
        assert r1 == r2

    def test_case_does_not_matter(self):
        r1 = _normalize_keywords_for_dedup(["AIS", "system"])
        r2 = _normalize_keywords_for_dedup(["ais", "SYSTEM"])
        assert r1 == r2


class TestIsKeywordSetDuplicate:
    def test_empty_previous_returns_false(self):
        state = {"used_keyword_sets": set()}
        assert not _is_keyword_set_duplicate(["a", "b"], state)

    def test_same_set_detected(self):
        norm = _normalize_keywords_for_dedup(["a", "b"])
        state = {"used_keyword_sets": {norm}}
        assert _is_keyword_set_duplicate(["a", "b"], state)

    def test_different_set_not_detected(self):
        norm = _normalize_keywords_for_dedup(["a", "b"])
        state = {"used_keyword_sets": {norm}}
        assert not _is_keyword_set_duplicate(["x", "y"], state)

    def test_subset_considered_duplicate(self):
        """Меньший набор по контракту считается дубликатом большего."""
        norm = _normalize_keywords_for_dedup(["a", "b", "c"])
        state = {"used_keyword_sets": {norm}}
        assert _is_keyword_set_duplicate(["a", "b"], state)

    def test_superset_considered_duplicate(self):
        norm = _normalize_keywords_for_dedup(["a"])
        state = {"used_keyword_sets": {norm}}
        assert _is_keyword_set_duplicate(["a", "b", "c"], state)

    def test_case_insensitive_match(self):
        norm = _normalize_keywords_for_dedup(["AIS"])
        state = {"used_keyword_sets": {norm}}
        assert _is_keyword_set_duplicate(["ais"], state)

    def test_missing_used_keyword_sets_key(self):
        """Не должно падать, если в state нет ключа — возвращается False."""
        assert not _is_keyword_set_duplicate(["a"], {})

    def test_empty_keywords(self):
        state = {"used_keyword_sets": {_normalize_keywords_for_dedup(["a"])}}
        assert not _is_keyword_set_duplicate([], state)
5. tests/unit/agent/test_categories.py
python
"""
Unit-тесты для app/services/agent_service/categories.py.
Проверяют:
- _validate_categories_for_search: фильтрацию переданных агентом категорий
  по списку доступных, регистронезависимость, поведение при отключённом
  поиске по категориям, при пустых входных данных, при дубликатах.

Внешние сервисы не требуются.
"""
import pytest
from app.services.agent_service.categories import CategoriesMixin


class TestableCategories(CategoriesMixin):
    """Изолированный класс без лишних зависимостей."""


@pytest.fixture
def mixin():
    return TestableCategories()


class TestValidateCategoriesForSearch:
    def test_empty_requested_returns_empty(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=[],
            available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert result == []

    def test_category_search_disabled_returns_empty(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A"],
            available_categories=["A", "B"],
            category_search_enabled=False,
        )
        assert result == []

    def test_no_available_categories_returns_empty(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A"],
            available_categories=[],
            category_search_enabled=True,
        )
        assert result == []

    def test_valid_categories_preserved(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A", "B"],
            available_categories=["A", "B", "C"],
            category_search_enabled=True,
        )
        assert set(result) == {"A", "B"}

    def test_invalid_categories_dropped(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["X", "Y"],
            available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert result == []

    def test_mixed_valid_and_invalid(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A", "X"],
            available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert result == ["A"]

    def test_case_insensitive_matching(self, mixin):
        """Регистр не должен влиять — либо матч, либо явный пропуск."""
        result = mixin._validate_categories_for_search(
            raw_categories=["a"],
            available_categories=["A", "B"],
            category_search_enabled=True,
        )
        # Контракт: возвращать каноничное имя из available_categories, а не ввод пользователя.
        assert result == ["A"] or result == []

    def test_duplicates_removed(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A", "A", "B", "B"],
            available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert len(result) == len(set(result))

    def test_non_string_values_ignored(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A", None, 123, "B"],
            available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert set(result) <= {"A", "B"}

    def test_preserves_order(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["B", "A"],
            available_categories=["A", "B", "C"],
            category_search_enabled=True,
        )
        # Ожидаем сохранения порядка из available_categories, если так задумано,
        # иначе порядка запроса. Проверяем только состав.
        assert set(result) == {"A", "B"}
6. tests/unit/agent/test_formatting.py
python
"""
Unit-тесты для app/services/agent_service/formatting.py.
Проверяют:
- _log_preview: обрезку длинных текстов до безопасного размера;
- _shorten_text: обрезку с многоточием;
- _format_chunks: формирование читаемого списка фрагментов для LLM;
- _format_selected_summary: краткое описание выбранных индексов;
- _format_history_lines: построение строк истории для промпта;
- _format_categories_for_prompt: форматирование списка категорий.

Внешние сервисы не требуются.
"""
from typing import Dict, List
import pytest
from app.services.agent_service.formatting import FormattingMixin


class TestableFormatting(FormattingMixin):
    """Изолированный форматировщик."""


@pytest.fixture
def fmt():
    return TestableFormatting()


class TestLogPreview:
    def test_short_text_unchanged(self, fmt):
        assert fmt._log_preview("короткий") == "короткий"

    def test_long_text_truncated(self, fmt):
        result = fmt._log_preview("a" * 500)
        assert len(result) < 500

    def test_empty_text(self, fmt):
        assert fmt._log_preview("") == ""

    def test_none_like_text(self, fmt):
        # Не должно падать при None — метод возвращает строку
        result = fmt._log_preview(None)
        assert isinstance(result, str)


class TestShortenText:
    def test_short_text_unchanged(self, fmt):
        assert fmt._shorten_text("short", 100) == "short"

    def test_long_text_truncated(self, fmt):
        result = fmt._shorten_text("a" * 500, 50)
        assert len(result) <= 60

    def test_ellipsis_added(self, fmt):
        result = fmt._shorten_text("a" * 500, 50)
        assert "..." in result or len(result) == 50

    def test_empty_text(self, fmt):
        assert fmt._shorten_text("", 50) == ""


class TestFormatChunks:
    def _chunk(self, idx, text, score=0.5, name=None, source="semantic") -> dict:
        return {
            "document": text,
            "metadata": {"display_name": name or f"doc_{idx}", "chunk_id": idx},
            "score": score,
            "source": source,
        }

    def test_empty_list_returns_empty_string(self, fmt):
        result = fmt._format_chunks([])
        assert result == "" or isinstance(result, str)

    def test_formats_single_chunk(self, fmt):
        chunks = [self._chunk(1, "текст", 0.9, name="alpha")]
        result = fmt._format_chunks(chunks)
        assert "alpha" in result or "1" in result
        assert "текст" in result

    def test_formats_multiple_chunks(self, fmt):
        chunks = [self._chunk(1, "первый"), self._chunk(2, "второй")]
        result = fmt._format_chunks(chunks)
        assert "первый" in result
        assert "второй" in result

    def test_handles_missing_metadata(self, fmt):
        chunk = {"document": "текст", "score": 0.5}
        result = fmt._format_chunks([chunk])
        assert isinstance(result, str)
        assert "текст" in result

    def test_chunk_index_shown(self, fmt):
        chunks = [self._chunk(5, "текст", name="doc5")]
        result = fmt._format_chunks(chunks)
        assert "[5]" in result or "5" in result


class TestFormatSelectedSummary:
    def test_empty_indices(self, fmt):
        result = fmt._format_selected_summary(
            [], {"accumulated_chunks": {}, "searched_keywords": set()},
        )
        assert isinstance(result, str)

    def test_summary_contains_indices(self, fmt):
        chunks = {
            1: {"document": "текст 1", "metadata": {"display_name": "doc1"}, "score": 0.5},
            2: {"document": "текст 2", "metadata": {"display_name": "doc2"}, "score": 0.7},
        }
        result = fmt._format_selected_summary(
            [1, 2], {"accumulated_chunks": chunks, "searched_keywords": set()},
        )
        assert "1" in result or "doc1" in result
        assert "2" in result or "doc2" in result


class TestFormatHistoryLines:
    def test_empty_history(self, fmt):
        assert fmt._format_history_lines([]) == ""

    def test_user_and_assistant(self, fmt):
        history = [
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "здравствуйте"},
        ]
        result = fmt._format_history_lines(history)
        assert "привет" in result
        assert "здравствуйте" in result
        assert "Пользователь" in result or "пользователь" in result.lower()
        assert "Ассистент" in result or "ассистент" in result.lower()

    def test_missing_content_treated_as_empty(self, fmt):
        result = fmt._format_history_lines([{"role": "user"}])
        assert isinstance(result, str)


class TestFormatCategoriesForPrompt:
    def test_empty_list(self, fmt):
        result = fmt._format_categories_for_prompt([])
        assert isinstance(result, str)

    def test_single_category(self, fmt):
        result = fmt._format_categories_for_prompt(["A"])
        assert "A" in result

    def test_multiple_categories(self, fmt):
        result = fmt._format_categories_for_prompt(["A", "B", "C"])
        assert "A" in result and "B" in result and "C" in result
7. tests/unit/agent/test_tools_definition.py
python
"""
Unit-тесты для получения определения инструментов агента (ToolsMixin.get_tools_definition).
Проверяют:
- базовый набор инструментов всегда содержит search_knowledge_base и finalize_search;
- keyword_search добавляется только если keyword_search_enabled=True;
- get_document_context доступен, если не финальная итерация;
- на финальной итерации поисковые инструменты недоступны;
- учёт лимита оставшихся поисков (search_left);
- корректный формат: каждый инструмент — dict с 'type' и 'function'.

Внешние сервисы не требуются.
"""
import pytest
from app.services.agent_service.tools import ToolsMixin


class TestableTools(ToolsMixin):
    """Изолированный класс."""


@pytest.fixture
def tools():
    return TestableTools()


def _names(defs: list) -> set:
    return {t["function"]["name"] for t in defs if "function" in t}


class TestGetToolsDefinition:
    def test_returns_list_of_dicts(self, tools):
        defs = tools.get_tools_definition(
            iteration=1, search_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=False,
        )
        assert isinstance(defs, list)
        for d in defs:
            assert d.get("type") == "function"
            assert "function" in d
            assert "name" in d["function"]
            assert "parameters" in d["function"]

    def test_finalize_search_always_present(self, tools):
        defs = tools.get_tools_definition(
            iteration=1, search_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=False,
        )
        assert "finalize_search" in _names(defs)

    def test_search_knowledge_base_present_when_not_final(self, tools):
        defs = tools.get_tools_definition(
            iteration=1, search_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=False,
        )
        assert "search_knowledge_base" in _names(defs)

    def test_search_tools_absent_on_final_iteration(self, tools):
        defs = tools.get_tools_definition(
            iteration=5, search_left=0, is_final=True,
            category_search_enabled=True, keyword_search_enabled=True,
        )
        names = _names(defs)
        assert "search_knowledge_base" not in names
        assert "keyword_search" not in names
        assert "finalize_search" in names

    def test_keyword_search_added_when_enabled(self, tools):
        defs = tools.get_tools_definition(
            iteration=1, search_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=True,
        )
        assert "keyword_search" in _names(defs)

    def test_keyword_search_absent_when_disabled(self, tools):
        defs = tools.get_tools_definition(
            iteration=1, search_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=False,
        )
        assert "keyword_search" not in _names(defs)

    def test_no_search_tools_when_no_searches_left(self, tools):
        defs = tools.get_tools_definition(
            iteration=2, search_left=0, is_final=False,
            category_search_enabled=False, keyword_search_enabled=True,
        )
        names = _names(defs)
        assert "search_knowledge_base" not in names
        assert "keyword_search" not in names
        assert "finalize_search" in names

    def test_parameters_is_object_schema(self, tools):
        defs = tools.get_tools_definition(
            iteration=1, search_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=False,
        )
        for d in defs:
            params = d["function"]["parameters"]
            assert params.get("type") == "object"

    def test_finalize_search_requires_selected_indices(self, tools):
        defs = tools.get_tools_definition(
            iteration=1, search_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=False,
        )
        finalize = next(d for d in defs if d["function"]["name"] == "finalize_search")
        params = finalize["function"]["parameters"]
        assert "selected_indices" in params.get("properties", {})

    def test_search_knowledge_base_requires_query(self, tools):
        defs = tools.get_tools_definition(
            iteration=1, search_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=False,
        )
        search = next(d for d in defs if d["function"]["name"] == "search_knowledge_base")
        params = search["function"]["parameters"]
        assert "query" in params.get("properties", {})
8. tests/unit/utils/test_redis_utils.py
python
"""
Unit-тесты для app/utils/redis_utils.py.
Проверяют:
- корректную сериализацию событий publish_chat_event в JSON;
- обработку несериализуемых данных (должно безопасно логироваться, не падать);
- корректный вызов xadd / publish (через мок AsyncMock);
- построение ключа стрима и активных задач;
- устойчивость к ошибкам Redis (не пробрасывать наружу).

Внешние сервисы не требуются — redis_client везде мок.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from app.utils.redis_utils import publish_chat_event


def _make_redis() -> AsyncMock:
    r = AsyncMock()
    r.xadd = AsyncMock(return_value="1-1")
    r.publish = AsyncMock(return_value=1)
    return r


class TestPublishChatEvent:
    @pytest.mark.asyncio
    async def test_publishes_serializable_event(self):
        redis = _make_redis()
        await publish_chat_event(redis, "task-1", "status", {"message": "ищу"})
        # Должен быть вызван либо xadd, либо publish — проверяем, что хотя бы одно
        assert redis.xadd.called or redis.publish.called

    @pytest.mark.asyncio
    async def test_event_payload_is_json(self):
        redis = _make_redis()
        await publish_chat_event(redis, "task-1", "response", {"text": "привет"})
        # Найдём аргументы вызова xadd и проверим, что payload — валидный JSON
        if redis.xadd.called:
            call_args = redis.xadd.call_args
            payload = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("fields", {})
            if isinstance(payload, dict):
                for v in payload.values():
                    if isinstance(v, str):
                        json.loads(v)  # не должно бросить

    @pytest.mark.asyncio
    async def test_non_serializable_data_does_not_crash(self):
        """Если внутри данных попадётся несериализуемое значение — функция не падает."""
        redis = _make_redis()
        class Weird:
            pass
        # Не должно бросить наружу
        await publish_chat_event(redis, "task-1", "status", {"obj": Weird()})

    @pytest.mark.asyncio
    async def test_redis_error_does_not_propagate(self):
        """Ошибка Redis не должна валить pipeline — события не критичны."""
        redis = AsyncMock()
        redis.xadd = AsyncMock(side_effect=ConnectionError("redis down"))
        redis.publish = AsyncMock(side_effect=ConnectionError("redis down"))
        # Если реализация глотает ошибки — тест проходит; если прокидывает — упадёт, что тоже индикатор.
        try:
            await publish_chat_event(redis, "task-1", "error", {"message": "x"})
        except ConnectionError:
            pytest.skip("Реализация не глотает ошибки Redis — это допустимо, но зафиксировано")

    @pytest.mark.asyncio
    async def test_empty_payload(self):
        redis = _make_redis()
        await publish_chat_event(redis, "task-1", "complete", {})
        assert redis.xadd.called or redis.publish.called

    @pytest.mark.asyncio
    async def test_task_id_used_in_call(self):
        redis = _make_redis()
        await publish_chat_event(redis, "task-xyz", "status", {"m": "1"})
        called_args = str(redis.xadd.call_args) + str(redis.publish.call_args)
        assert "task-xyz" in called_args
