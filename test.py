    async def _should_compress_history(self, history: List[Dict]) -> bool:
        """
        Решение о сжатии:
        - мало сообщений (< MIN_MESSAGES), но много токенов (> MAX_TOKENS) -> сжимать;
        - много сообщений (> MIN_MESSAGES) -> сжимать независимо от токенов;
        - совсем короткая история (< 2 сообщений) -> не сжимать.
        """
        if not history:
            return False

        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")[: settings.MAX_HISTORY_MESSAGE_CHARS]}
            for m in history
            if m.get("role") in ("user", "assistant")
        ]
        if len(messages) < 2:
            return False

        # Порог по числу сообщений — независимо от токенов
        if len(messages) > settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES:
            logger.debug(
                f"History: {len(messages)} msgs > MIN_MESSAGES="
                f"{settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES} -> compress"
            )
            return True

        # Порог по токенам — для случая «мало сообщений, но они длинные»
        try:
            tokens = await count_messages_tokens_async(
                messages=messages, tools=None, use_exact=True,
            )
        except Exception as e:
            logger.warning(f"Не удалось посчитать токены истории: {e}")
            return False

        needs = tokens > settings.AGENT_HISTORY_COMPRESS_MAX_TOKENS
        logger.debug(
            f"History: {len(messages)} msgs, {tokens} tokens, "
            f"MAX_TOKENS={settings.AGENT_HISTORY_COMPRESS_MAX_TOKENS} -> "
            f"compress={needs}"
        )
        return needs

--------


@pytest.mark.asyncio
@patch("app.services.agent_service.history.count_messages_tokens_async", new_callable=AsyncMock)
@patch("app.services.agent_service.history.settings")
async def test_should_compress_by_tokens_when_messages_few(mock_settings, mock_count, mixin):
    """Мало сообщений, но много токенов -> всё равно сжимаем."""
    mock_settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES = 100
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_TOKENS = 2000
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 1500
    mock_count.return_value = 5000
    assert await mixin._should_compress_history(_make_history(20))


@pytest.mark.asyncio
@patch("app.services.agent_service.history.count_messages_tokens_async", new_callable=AsyncMock)
@patch("app.services.agent_service.history.settings")
async def test_should_not_compress_when_below_both_thresholds(mock_settings, mock_count, mixin):
    """Мало сообщений и мало токенов -> не сжимаем."""
    mock_settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES = 100
    mock_settings.AGENT_HISTORY_COMPRESS_MAX_TOKENS = 2000
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 1500
    mock_count.return_value = 500
    assert not await mixin._should_compress_history(_make_history(20))


@pytest.mark.asyncio
@patch("app.services.agent_service.history.settings")
async def test_should_not_compress_single_message(mock_settings, mixin):
    """Одно сообщение — сжимать нечего."""
    mock_settings.AGENT_HISTORY_COMPRESS_MIN_MESSAGES = 12
    mock_settings.MAX_HISTORY_MESSAGE_CHARS = 1500
    assert not await mixin._should_compress_history(_make_history(1))
