
"""
Unit-тесты для app/services/agent_service/categories.py.

Проверяют:
- _validate_categories_for_search: возврат None, когда фильтр не применим
  (category_search выключен, категории пусты, available_categories пуст,
  все категории отклонены), и List с каноничными именами при валидных
  входных данных; регистронезависимость; дедупликацию; поведение при
  AGENT_CATEGORY_VALIDATION_ENABLED=False.
- _format_categories_for_prompt: порядок (корневые → вложенные), обрезку
  до AGENT_CATEGORY_PROMPT_LIMIT, наличие хвостовой подсказки про опциональность.

Внешние сервисы не требуются.
"""
from unittest.mock import patch
import pytest
from app.services.agent_service.categories import CategoriesMixin


class TestableCategories(CategoriesMixin):
    """Изолированный класс без лишних зависимостей."""
    pass


@pytest.fixture
def mixin():
    return TestableCategories()


class TestValidateCategoriesForSearch:
    def test_empty_requested_returns_none(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=[], available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert result is None

    def test_none_requested_returns_none(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=None, available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert result is None

    def test_category_search_disabled_returns_none(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A"], available_categories=["A", "B"],
            category_search_enabled=False,
        )
        assert result is None

    def test_no_available_categories_returns_none(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A"], available_categories=[],
            category_search_enabled=True,
        )
        assert result is None

    def test_valid_categories_returned_as_list(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A", "B"],
            available_categories=["A", "B", "C"],
            category_search_enabled=True,
        )
        assert set(result) == {"A", "B"}

    def test_all_invalid_returns_none(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["X", "Y"], available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert result is None

    def test_mixed_valid_and_invalid_returns_only_valid(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A", "X"], available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert result == ["A"]

    def test_case_insensitive_matching_returns_canonical(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["a"], available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert result == ["A"]

    def test_duplicates_removed(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A", "A", "B", "B"],
            available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert len(result) == len(set(result))
        assert set(result) == {"A", "B"}

    def test_non_string_values_converted_and_filtered(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["A", None, 123, "B"],
            available_categories=["A", "B"],
            category_search_enabled=True,
        )
        # None -> "none" (не матчится), 123 -> "123" (не матчится),
        # остаются только A и B
        assert set(result) == {"A", "B"}

    def test_whitespace_stripped(self, mixin):
        result = mixin._validate_categories_for_search(
            raw_categories=["  A  "], available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert result == ["A"]

    @patch("app.services.agent_service.categories.settings")
    def test_validation_disabled_passes_raw(self, mock_settings, mixin):
        mock_settings.AGENT_CATEGORY_VALIDATION_ENABLED = False
        result = mixin._validate_categories_for_search(
            raw_categories=["Unknown"], available_categories=["A", "B"],
            category_search_enabled=True,
        )
        assert result == ["Unknown"]

    @patch("app.services.agent_service.categories.settings")
    def test_validation_disabled_empty_returns_none(self, mock_settings, mixin):
        mock_settings.AGENT_CATEGORY_VALIDATION_ENABLED = False
        result = mixin._validate_categories_for_search(
            raw_categories=["", "  "], available_categories=["A"],
            category_search_enabled=True,
        )
        assert result is None


class TestFormatCategoriesForPrompt:
    def test_empty_list_returns_empty_string(self, mixin):
        assert mixin._format_categories_for_prompt([]) == ""

    @patch("app.services.agent_service.categories.settings")
    def test_root_before_nested(self, mock_settings, mixin):
        mock_settings.AGENT_CATEGORY_PROMPT_LIMIT = 10
        result = mixin._format_categories_for_prompt(
            ["doc:sub", "root", "doc:another"]
        )
        # "root" должен быть до вложенных
        assert result.index("root") < result.index("doc:sub")

    @patch("app.services.agent_service.categories.settings")
    def test_respects_prompt_limit(self, mock_settings, mixin):
        mock_settings.AGENT_CATEGORY_PROMPT_LIMIT = 3
        cats = [f"cat_{i}" for i in range(10)]
        result = mixin._format_categories_for_prompt(cats)
        # Должно быть упоминание "и ещё N"
        assert "ещё" in result

    @patch("app.services.agent_service.categories.settings")
    def test_mentions_optionality(self, mock_settings, mixin):
        mock_settings.AGENT_CATEGORY_PROMPT_LIMIT = 10
        result = mixin._format_categories_for_prompt(["A"])
        assert "НЕОБЯЗАТЕЛЬНЫЙ" in result or "необязательный" in result.lower()


"""
Unit-тесты для app/services/agent_service/formatting.py.

Проверяют:
- _shorten_text: обрезку с суффиксом "...[фрагмент обрезан]";
- _log_preview: сжатие пробелов и обрезку до AGENT_LOG_PREVIEW_CHARS;
- _format_history_lines: возврат единой строки, удаление "/no_think",
  пропуск пустых сообщений, роли Пользователь/Ассистент;
- _chunk_kw_hits: регистронезависимое совпадение;
- _chunk_short_label: компактная метка [idx]name(source,kw+/kw-);
- _format_selected_summary(selected, state): одну строку с метками чанков;
- _format_chunks: заголовок с chunk_id (в т.ч. строкой "1,2,3") и matched_keywords.

Внешние сервисы не требуются.
"""
from unittest.mock import patch
import pytest
from app.services.agent_service.formatting import FormattingMixin


class TestableFormatting(FormattingMixin):
    """Изолированный форматировщик."""
    pass


@pytest.fixture
def fmt():
    return TestableFormatting()


class TestShortenText:
    def test_short_text_unchanged(self, fmt):
        assert fmt._shorten_text("short", 100) == "short"

    def test_exact_limit_unchanged(self, fmt):
        assert fmt._shorten_text("x" * 50, 50) == "x" * 50

    def test_long_text_truncated_with_suffix(self, fmt):
        result = fmt._shorten_text("a" * 500, 50)
        assert len(result) > 50
        assert "фрагмент обрезан" in result
        assert result.startswith("a" * 50)

    def test_empty_text(self, fmt):
        assert fmt._shorten_text("", 50) == ""

    def test_none_text(self, fmt):
        assert fmt._shorten_text(None, 50) == ""


class TestLogPreview:
    @patch("app.services.agent_service.formatting.settings")
    def test_short_text_unchanged(self, mock_settings, fmt):
        mock_settings.AGENT_LOG_PREVIEW_CHARS = 100
        assert fmt._log_preview("короткий") == "короткий"

    @patch("app.services.agent_service.formatting.settings")
    def test_long_text_truncated(self, mock_settings, fmt):
        mock_settings.AGENT_LOG_PREVIEW_CHARS = 10
        result = fmt._log_preview("a" * 500)
        assert len(result) <= 13  # 10 символов + "..."
        assert result.endswith("...")

    @patch("app.services.agent_service.formatting.settings")
    def test_whitespace_collapsed(self, mock_settings, fmt):
        mock_settings.AGENT_LOG_PREVIEW_CHARS = 100
        result = fmt._log_preview("a\n\n  b\t\tc")
        assert result == "a b c"

    def test_empty_text(self, fmt):
        assert fmt._log_preview("") == ""

    def test_none_text(self, fmt):
        assert fmt._log_preview(None) == ""


class TestFormatHistoryLines:
    def test_empty_history(self, fmt):
        assert fmt._format_history_lines([]) == ""

    def test_user_and_assistant_string(self, fmt):
        history = [
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "здравствуйте"},
        ]
        result = fmt._format_history_lines(history)
        assert isinstance(result, str)
        assert "Пользователь: привет" in result
        assert "Ассистент: здравствуйте" in result
        assert "\n" in result

    def test_no_think_stripped(self, fmt):
        history = [{"role": "user", "content": "/no_think привет"}]
        result = fmt._format_history_lines(history)
        assert "/no_think" not in result
        assert "привет" in result

    def test_empty_content_skipped(self, fmt):
        history = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "текст"},
        ]
        result = fmt._format_history_lines(history)
        # Пустое сообщение не должно давать строку "Пользователь: "
        assert result.count("Пользователь:") == 1

    def test_missing_role_treated_as_assistant(self, fmt):
        history = [{"content": "без роли"}]
        result = fmt._format_history_lines(history)
        assert "Ассистент" in result


class TestChunkKwHits:
    def test_no_keywords(self, fmt):
        assert not fmt._chunk_kw_hits("текст", set())

    def test_no_text(self, fmt):
        assert not fmt._chunk_kw_hits("", {"аис"})

    def test_match_returns_true(self, fmt):
        assert fmt._chunk_kw_hits("Руководство АИС", {"аис"})

    def test_case_insensitive(self, fmt):
        assert fmt._chunk_kw_hits("руководство АИС", {"аис"})

    def test_no_match(self, fmt):
        assert not fmt._chunk_kw_hits("случайный текст", {"аис"})


class TestChunkShortLabel:
    def _chunk(self, doc, score=0.5, source="semantic", name="doc1"):
        return {
            "document": doc, "score": score, "source": source,
            "metadata": {"display_name": name},
        }

    def test_kw_plus_when_match(self, fmt):
        chunk = self._chunk("текст с АИС")
        label = fmt._chunk_short_label(5, chunk, {"аис"})
        assert "[5]" in label
        assert "doc1" in label
        assert "kw+" in label

    def test_kw_minus_when_no_match(self, fmt):
        chunk = self._chunk("другой текст")
        label = fmt._chunk_short_label(5, chunk, {"аис"})
        assert "kw-" in label

    def test_missing_metadata(self, fmt):
        chunk = {"document": "x", "score": 0.5, "source": "semantic"}
        label = fmt._chunk_short_label(1, chunk, set())
        assert "unknown" in label


class TestFormatSelectedSummary:
    def _state(self, chunks, keywords=None):
        return {
            "accumulated_chunks": chunks,
            "searched_keywords": keywords or set(),
        }

    def test_empty_selection(self, fmt):
        result = fmt._format_selected_summary([], self._state({}))
        assert result == "пусто"

    def test_missing_indices_ignored(self, fmt):
        state = self._state({1: {"document": "x", "metadata": {"display_name": "d"}}})
        result = fmt._format_selected_summary([1, 99], state)
        assert "d" in result
        assert "пусто" not in result

    def test_multiple_indices(self, fmt):
        state = self._state({
            1: {"document": "doc a", "metadata": {"display_name": "A"}, "score": 0.5},
            2: {"document": "doc b", "metadata": {"display_name": "B"}, "score": 0.7},
        })
        result = fmt._format_selected_summary([1, 2], state)
        assert "A" in result and "B" in result
        assert ", " in result


class TestFormatChunks:
    def test_empty_returns_placeholder(self, fmt):
        assert fmt._format_chunks([]) == "Нет фрагментов."

    def test_single_chunk(self, fmt):
        chunks = [{
            "index": 1, "display_name": "doc1", "chunk_id": 5,
            "relevance": 0.9, "text": "текст",
        }]
        result = fmt._format_chunks(chunks)
        assert "[1]" in result
        assert "doc1" in result
        assert "текст" in result

    def test_matched_keywords_note(self, fmt):
        chunks = [{
            "index": 1, "display_name": "doc1", "chunk_id": 5,
            "relevance": 0.9, "text": "текст",
            "matched_keywords": ["аис", "система"],
        }]
        result = fmt._format_chunks(chunks)
        assert "содержит слова" in result
        assert "аис" in result and "система" in result

    def test_merged_chunk_string_id(self, fmt):
        chunks = [{
            "index": 3, "display_name": "doc", "chunk_id": "1,2,3",
            "relevance": 0.5, "text": "merged",
        }]
        result = fmt._format_chunks(chunks)
        assert "чанки 1,2,3" in result

    def test_multiple_chunks_separator(self, fmt):
        chunks = [
            {"index": 1, "display_name": "a", "chunk_id": 1, "relevance": 0.5, "text": "x"},
            {"index": 2, "display_name": "b", "chunk_id": 2, "relevance": 0.6, "text": "y"},
        ]
        result = fmt._format_chunks(chunks)
        assert "\n---\n" in result


"""
Unit-тесты для app/services/agent_service/keyword_dedup.py.

Проверяют:
- _normalize_keywords_for_dedup: разбивку фраз на слова, lower,
  strip пунктуации, отбрасывание коротких (<2 симв.) слов,
  возврат frozenset (порядок не важен);
- _is_keyword_set_duplicate: точное совпадение, подмножество,
  надмножество уже использованного набора; регистронезависимость;
  безопасность на пустых наборах.

Внешние сервисы не требуются.
"""
import pytest
from app.services.agent_service.keyword_dedup import KeywordDedupMixin


class TestableDedup(KeywordDedupMixin):
    """Изолированный миксин."""
    pass


@pytest.fixture
def mixin():
    return TestableDedup()


class TestNormalizeKeywords:
    def test_returns_frozenset(self, mixin):
        result = mixin._normalize_keywords_for_dedup(["a"])
        assert isinstance(result, frozenset)

    def test_empty_list_returns_empty_frozenset(self, mixin):
        assert mixin._normalize_keywords_for_dedup([]) == frozenset()

    def test_lowercase(self, mixin):
        assert mixin._normalize_keywords_for_dedup(["ABC"]) == frozenset({"abc"})

    def test_multiword_phrase_split(self, mixin):
        result = mixin._normalize_keywords_for_dedup(["руководство аис"])
        assert result == frozenset({"руководство", "аис"})

    def test_multiword_matches_separate_words(self, mixin):
        r1 = mixin._normalize_keywords_for_dedup(["руководство аис"])
        r2 = mixin._normalize_keywords_for_dedup(["аис", "руководство"])
        assert r1 == r2

    def test_punctuation_stripped(self, mixin):
        result = mixin._normalize_keywords_for_dedup(["аис,", "(система)"])
        assert result == frozenset({"аис", "система"})

    def test_short_words_dropped(self, mixin):
        result = mixin._normalize_keywords_for_dedup(["a", "аb", "c"])
        assert "a" not in result
        assert "c" not in result
        assert "аb" in result

    def test_duplicates_removed(self, mixin):
        result = mixin._normalize_keywords_for_dedup(["АИС", "аис", "ais"])
        assert result == frozenset({"аис", "ais"})

    def test_non_string_values_handled(self, mixin):
        # Метод использует str(kw) — не должно падать
        result = mixin._normalize_keywords_for_dedup([123, None])
        assert isinstance(result, frozenset)


class TestIsKeywordSetDuplicate:
    def test_empty_previous_returns_false(self, mixin):
        assert not mixin._is_keyword_set_duplicate(["a", "b"], {"used_keyword_sets": set()})

    def test_same_set_detected(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["a", "b"])
        state = {"used_keyword_sets": {norm}}
        assert mixin._is_keyword_set_duplicate(["a", "b"], state)

    def test_case_insensitive_match(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["AIS"])
        state = {"used_keyword_sets": {norm}}
        assert mixin._is_keyword_set_duplicate(["ais"], state)

    def test_different_set_not_detected(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["a", "b"])
        state = {"used_keyword_sets": {norm}}
        assert not mixin._is_keyword_set_duplicate(["x", "y"], state)

    def test_subset_detected(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["a", "b", "c"])
        state = {"used_keyword_sets": {norm}}
        assert mixin._is_keyword_set_duplicate(["a", "b"], state)

    def test_superset_detected(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["a"])
        state = {"used_keyword_sets": {norm}}
        assert mixin._is_keyword_set_duplicate(["a", "b", "c"], state)

    def test_empty_keywords_returns_false(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["a"])
        state = {"used_keyword_sets": {norm}}
        assert not mixin._is_keyword_set_duplicate([], state)

    def test_missing_used_keyword_sets_key(self, mixin):
        assert not mixin._is_keyword_set_duplicate(["a"], {})

    def test_phrase_matches_words(self, mixin):
        """Фраза "руководство аис" должна совпасть с ["аис", "руководство"]."""
        norm = mixin._normalize_keywords_for_dedup(["аис", "руководство"])
        state = {"used_keyword_sets": {norm}}
        assert mixin._is_keyword_set_duplicate(["руководство аис"], state)


"""
Unit-тесты для app/services/agent_service/tools.py (get_tools_definition).

Проверяют:
- на финальной итерации возвращается только finalize_search;
- в обычной итерации присутствуют keyword_search (если включён и есть
  попытки), search_knowledge_base (если есть попытки), get_document_context,
  finalize_search;
- keyword_search исчезает при keyword_search_enabled=False или
  search_calls_left=0;
- search_knowledge_base исчезает при search_calls_left=0;
- параметр categories добавляется в оба поисковых инструмента только
  при category_search_enabled=True;
- get_document_context требует fragment_indices (array of integer);
- каждая tool — dict с type=function, function.name, function.parameters.

Внешние сервисы не требуются.
"""
from unittest.mock import patch
import pytest
from app.services.agent_service.tools import ToolsMixin


class TestableTools(ToolsMixin):
    """Изолированный миксин."""
    pass


@pytest.fixture
def tools():
    return TestableTools()


def _names(defs):
    return [t["function"]["name"] for t in defs]


def _find(defs, name):
    for t in defs:
        if t["function"]["name"] == name:
            return t
    return None


class TestGetToolsDefinition:
    @patch("app.services.agent_service.tools.settings")
    def test_final_iteration_returns_only_finalize(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=5, search_calls_left=3, is_final=True,
            category_search_enabled=True, keyword_search_enabled=True,
        )
        assert _names(defs) == ["finalize_search"]

    @patch("app.services.agent_service.tools.settings")
    def test_normal_iteration_contains_core_tools(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=1, search_calls_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=True,
        )
        names = _names(defs)
        assert "keyword_search" in names
        assert "search_knowledge_base" in names
        assert "get_document_context" in names
        assert "finalize_search" in names

    @patch("app.services.agent_service.tools.settings")
    def test_keyword_search_absent_when_disabled(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=1, search_calls_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=False,
        )
        assert "keyword_search" not in _names(defs)
        assert "search_knowledge_base" in _names(defs)

    @patch("app.services.agent_service.tools.settings")
    def test_no_search_tools_when_no_calls_left(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=2, search_calls_left=0, is_final=False,
            category_search_enabled=False, keyword_search_enabled=True,
        )
        names = _names(defs)
        assert "keyword_search" not in names
        assert "search_knowledge_base" not in names
        assert "get_document_context" in names
        assert "finalize_search" in names

    @patch("app.services.agent_service.tools.settings")
    def test_categories_added_only_when_enabled(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=1, search_calls_left=3, is_final=False,
            category_search_enabled=True, keyword_search_enabled=True,
        )
        kw = _find(defs, "keyword_search")
        sem = _find(defs, "search_knowledge_base")
        assert "categories" in kw["function"]["parameters"]["properties"]
        assert "categories" in sem["function"]["parameters"]["properties"]

    @patch("app.services.agent_service.tools.settings")
    def test_categories_absent_when_disabled(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=1, search_calls_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=True,
        )
        kw = _find(defs, "keyword_search")
        sem = _find(defs, "search_knowledge_base")
        assert "categories" not in kw["function"]["parameters"]["properties"]
        assert "categories" not in sem["function"]["parameters"]["properties"]

    @patch("app.services.agent_service.tools.settings")
    def test_finalize_requires_selected_indices(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=1, search_calls_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=False,
        )
        finalize = _find(defs, "finalize_search")
        params = finalize["function"]["parameters"]
        assert "selected_indices" in params["properties"]
        assert params["required"] == ["selected_indices"]

    @patch("app.services.agent_service.tools.settings")
    def test_search_requires_query(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=1, search_calls_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=False,
        )
        search = _find(defs, "search_knowledge_base")
        params = search["function"]["parameters"]
        assert "query" in params["properties"]
        assert params["required"] == ["query"]

    @patch("app.services.agent_service.tools.settings")
    def test_keyword_search_requires_keywords(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=1, search_calls_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=True,
        )
        kw = _find(defs, "keyword_search")
        params = kw["function"]["parameters"]
        assert "keywords" in params["properties"]
        assert params["required"] == ["keywords"]

    @patch("app.services.agent_service.tools.settings")
    def test_get_document_context_uses_fragment_indices(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=1, search_calls_left=3, is_final=False,
            category_search_enabled=False, keyword_search_enabled=True,
        )
        ctx = _find(defs, "get_document_context")
        params = ctx["function"]["parameters"]
        assert "fragment_indices" in params["properties"]
        assert params["properties"]["fragment_indices"]["type"] == "array"
        assert params["required"] == ["fragment_indices"]

    @patch("app.services.agent_service.tools.settings")
    def test_all_tools_have_valid_schema(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=1, search_calls_left=3, is_final=False,
            category_search_enabled=True, keyword_search_enabled=True,
        )
        for t in defs:
            assert t["type"] == "function"
            fn = t["function"]
            assert "name" in fn and fn["name"]
            assert "description" in fn and fn["description"]
            assert fn["parameters"]["type"] == "object"
            assert "properties" in fn["parameters"]

    @patch("app.services.agent_service.tools.settings")
    def test_search_calls_left_in_description(self, mock_settings, tools):
        mock_settings.AGENT_KEYWORD_SEARCH_MAX_KEYWORDS = 3
        defs = tools.get_tools_definition(
            iteration=1, search_calls_left=7, is_final=False,
            category_search_enabled=False, keyword_search_enabled=True,
        )
        kw = _find(defs, "keyword_search")
        sem = _find(defs, "search_knowledge_base")
        assert "7" in kw["function"]["description"]
        assert "7" in sem["function"]["description"]


