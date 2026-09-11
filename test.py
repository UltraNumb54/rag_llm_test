
"""
Unit-тесты для SetupMixin (loop/setup.py).

Проверяют:
- _build_sliding_window_history: фильтр по роли ДО среза, лимит по числу сообщений,
  обрезку каждого сообщения по MAX_HISTORY_MESSAGE_CHARS;
- _prepare_agent_history: режимы full и compressed; вызов _should_compress_history
  и _compress_history через self (мокаются в TestableSetup);
- _load_categories: загрузка из Qdrant, обрезка по AGENT_CATEGORY_PROMPT_LIMIT;
- _build_initial_prompt: формирование стартового промпта, /no_think-префикс, категории.

Внешние сервисы замоканы. Тесты изолированы через TestableSetup.
_should_compress_history и _compress_history НЕ реализуются в тестовом классе —
они подставляются как AsyncMock в тестах, потому что в реальном AgentService
приходят из HistoryMixin, а не из SetupMixin.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from app.services.agent_service.loop.setup import SetupMixin


class TestableSetup(SetupMixin):
    """
    Изолированный SetupMixin. _should_compress_history и _compress_history
    живут в HistoryMixin и в проде приходят через AgentService; здесь они
    подставляются моками в каждом тесте, где нужны.
    """

    def __init__(self):
        # Заглушки методов из HistoryMixin, чтобы self.<...> не падал
        self._should_compress_history = AsyncMock(return_value=False)
        self._compress_history = AsyncMock(return_value="")

    def _format_history_lines(self, messages):
        lines = []
        for m in messages:
            role = "Пользователь" if m["role"] == "user" else "Ассистент"
            lines.append(f"{role}: {m.get('content', '')}")
        return "\n".join(lines)

    @property
    def agent_system_prompt(self):
        return "Ты агент."


@pytest.fixture
def setup():
    return TestableSetup()


def _make_history(n: int) -> list:
    """История из n пар (user + assistant) = 2n сообщений."""
    history = []
    for i in range(n):
        history.append({"role": "user", "content": f"Вопрос {i}"})
        history.append({"role": "assistant", "content": f"Ответ {i}"})
    return history


# ─── _build_sliding_window_history ───

class TestSlidingWindow:
    def test_empty_history(self, setup):
        assert setup._build_sliding_window_history([], logger) == ""

    @patch("app.services.agent_service.loop.setup.settings")
    def test_respects_max_history_limit(self, mock_settings, setup):
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 4
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = True
        history = _make_history(10)  # 20 элементов
        result = setup._build_sliding_window_history(history, logger)
        # [-4:] = [U8, A8, U9, A9]
        assert "Вопрос 9" in result
        assert "Ответ 9" in result
        assert "Вопрос 8" in result
        assert "Ответ 8" in result
        assert "Вопрос 0" not in result

    @patch("app.services.agent_service.loop.setup.settings")
    def test_excludes_assistant_filters_before_slice(self, mock_settings, setup):
        """AGENT_HISTORY_INCLUDE_ASSISTANT=False: фильтр ДО среза."""
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 2
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = False
        history = _make_history(10)
        result = setup._build_sliding_window_history(history, logger)
        # 2 последних USER-сообщения: Вопрос 8, Вопрос 9
        assert "Вопрос 9" in result
        assert "Вопрос 8" in result
        assert "Ассистент" not in result
        assert "Ответ" not in result

    @patch("app.services.agent_service.loop.setup.settings")
    def test_excludes_assistant_with_limit_one(self, mock_settings, setup):
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 1
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = False
        history = _make_history(10)
        result = setup._build_sliding_window_history(history, logger)
        assert "Вопрос 9" in result
        assert "Вопрос 8" not in result
        assert "Ассистент" not in result

    @patch("app.services.agent_service.loop.setup.settings")
    def test_truncates_long_messages(self, mock_settings, setup):
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 10
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 10
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = True
        history = [{"role": "user", "content": "a" * 500}]
        result = setup._build_sliding_window_history(history, logger)
        assert len(result) < 50
        assert "a" * 10 in result

    @patch("app.services.agent_service.loop.setup.settings")
    def test_include_assistant_keeps_both_roles(self, mock_settings, setup):
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 10
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = True
        history = _make_history(3)
        result = setup._build_sliding_window_history(history, logger)
        assert "Пользователь" in result
        assert "Ассистент" in result

    @patch("app.services.agent_service.loop.setup.settings")
    def test_all_assistant_and_excluded_returns_empty(self, mock_settings, setup):
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 10
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = False
        history = [{"role": "assistant", "content": "только ответы"}]
        result = setup._build_sliding_window_history(history, logger)
        assert result == ""


# ─── _prepare_agent_history ───

class TestPrepareAgentHistory:

    @pytest.mark.asyncio
    async def test_empty_history_returns_empty(self, setup):
        text, compressed = await setup._prepare_agent_history(
            AsyncMock(), "full", [], "вопрос", logger,
        )
        assert text == ""
        assert compressed is False
        setup._should_compress_history.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.setup.settings")
    async def test_full_mode_uses_sliding_window(self, mock_settings, setup):
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 10
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = True
        history = _make_history(3)
        text, compressed = await setup._prepare_agent_history(
            AsyncMock(), "full", history, "вопрос", logger,
        )
        assert not compressed
        assert "Вопрос" in text
        setup._should_compress_history.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.setup.settings")
    async def test_compressed_short_history_uses_sliding_window(
        self, mock_settings, setup,
    ):
        """_should_compress_history вернул False -> fallback на sliding window."""
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 10
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = True
        setup._should_compress_history = AsyncMock(return_value=False)
        short_history = _make_history(1)
        text, compressed = await setup._prepare_agent_history(
            AsyncMock(), "compressed", short_history, "вопрос", logger,
        )
        assert not compressed
        assert text != ""
        setup._should_compress_history.assert_awaited_once()
        setup._compress_history.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.setup.settings")
    async def test_compressed_long_history_uses_compression(
        self, mock_settings, setup,
    ):
        """_should_compress_history=True -> _compress_history вызывается."""
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 10
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = True
        setup._should_compress_history = AsyncMock(return_value=True)
        setup._compress_history = AsyncMock(return_value="сжатая история")
        history = _make_history(10)
        text, compressed = await setup._prepare_agent_history(
            AsyncMock(), "compressed", history, "вопрос", logger,
        )
        assert compressed is True
        assert text == "сжатая история"
        setup._compress_history.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.setup.settings")
    async def test_compressed_fallback_when_compression_empty(
        self, mock_settings, setup,
    ):
        """_compress_history вернул пусто -> fallback на sliding window."""
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 10
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = True
        setup._should_compress_history = AsyncMock(return_value=True)
        setup._compress_history = AsyncMock(return_value="")
        history = _make_history(10)
        text, compressed = await setup._prepare_agent_history(
            AsyncMock(), "compressed", history, "вопрос", logger,
        )
        assert compressed is False
        assert "Вопрос" in text

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.setup.settings")
    async def test_compressed_passes_redis_client(self, mock_settings, setup):
        """redis_client прокидывается в _compress_history."""
        mock_settings.MAX_HISTORY_MESSAGES_FOR_CONTEXT = 10
        mock_settings.MAX_HISTORY_MESSAGE_CHARS = 500
        mock_settings.AGENT_HISTORY_INCLUDE_ASSISTANT = True
        setup._should_compress_history = AsyncMock(return_value=True)
        setup._compress_history = AsyncMock(return_value="ok")
        redis = MagicMock()
        history = _make_history(10)
        await setup._prepare_agent_history(
            AsyncMock(), "compressed", history, "вопрос", logger, redis_client=redis,
        )
        # Проверяем, что redis_client попал в вызов
        call_args = setup._compress_history.await_args
        assert redis in call_args.args or redis in call_args.kwargs.values()


# ─── _load_categories ───

class TestLoadCategories:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self, setup):
        qdrant = AsyncMock()
        result = await setup._load_categories(qdrant, False, "full", logger)
        assert result == []
        qdrant.get_categories_list.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.agent_service.loop.setup.settings")
    @patch("app.services.agent_service.loop.setup.record_agent_categories_loaded")
    async def test_loads_categories(self, mock_record, mock_settings, setup):
        mock_settings.AGENT_CATEGORY_PROMPT_LIMIT = 50
        qdrant = AsyncMock()
        qdrant.get_categories_list = AsyncMock(return_value=["A", "B", "C"])
        result = await setup._load_categories(qdrant, True, "full", logger)
        assert result == ["A", "B", "C"]
        mock_record.assert_called_once()


# ─── _build_initial_prompt ───

class TestBuildInitialPrompt:
    @patch("app.services.agent_service.loop.setup.settings")
    def test_no_categories_no_history(self, mock_settings, setup):
        mock_settings.AGENT_DISABLE_THINKING = False
        mock_settings.AGENT_USE_NO_THINK_PREFIX = False
        messages = setup._build_initial_prompt("вопрос", "", [], False)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "вопрос" in messages[1]["content"]
        assert "НЕТ фрагментов" in messages[1]["content"]

    @patch("app.services.agent_service.loop.setup.settings")
    def test_with_history(self, mock_settings, setup):
        mock_settings.AGENT_DISABLE_THINKING = False
        mock_settings.AGENT_USE_NO_THINK_PREFIX = False
        messages = setup._build_initial_prompt("вопрос", "история диалога", [], False)
        assert "история диалога" in messages[1]["content"]
        assert "История диалога" in messages[1]["content"]

    @patch("app.services.agent_service.loop.setup.settings")
    def test_with_categories(self, mock_settings, setup):
        mock_settings.AGENT_DISABLE_THINKING = False
        mock_settings.AGENT_USE_NO_THINK_PREFIX = False
        setup._format_categories_for_prompt = MagicMock(return_value="Категории: A, B\n")
        messages = setup._build_initial_prompt("вопрос", "", ["A", "B"], True)
        assert "Категории" in messages[1]["content"]

    @patch("app.services.agent_service.loop.setup.settings")
    def test_categories_disabled(self, mock_settings, setup):
        mock_settings.AGENT_DISABLE_THINKING = False
        mock_settings.AGENT_USE_NO_THINK_PREFIX = False
        setup._format_categories_for_prompt = MagicMock(return_value="Категории: A\n")
        messages = setup._build_initial_prompt("вопрос", "", ["A"], False)
        assert "Категории" not in messages[1]["content"]

    @patch("app.services.agent_service.loop.setup.settings")
    def test_no_think_prefix_added(self, mock_settings, setup):
        mock_settings.AGENT_DISABLE_THINKING = True
        mock_settings.AGENT_USE_NO_THINK_PREFIX = True
        messages = setup._build_initial_prompt("вопрос", "", [], False)
        assert messages[1]["content"].startswith("/no_think ")

    @patch("app.services.agent_service.loop.setup.settings")
    def test_no_think_prefix_not_added_when_disabled(self, mock_settings, setup):
        mock_settings.AGENT_DISABLE_THINKING = True
        mock_settings.AGENT_USE_NO_THINK_PREFIX = False
        messages = setup._build_initial_prompt("вопрос", "", [], False)
        assert not messages[1]["content"].startswith("/no_think ")
