
"""
Unit-тесты для app/services/agent_service/keyword_dedup.py.

Проверяют:
- _normalize_keywords_for_dedup: разбивку многословных фраз на отдельные
  слова, приведение к lower, strip пунктуации, отбрасывание слов короче
  2 символов, дедупликацию, возврат frozenset;
- _is_keyword_set_duplicate: точное совпадение, подмножество, надмножество
  уже использованного набора, регистронезависимость, безопасность на
  пустых наборах и отсутствующем ключе used_keyword_sets.

ВАЖНО: слова короче 2 символов молча отбрасываются нормализатором.
Тесты используют ключевые слова длиной >= 2 символов, кроме отдельного
теста test_short_words_dropped, который это поведение фиксирует.

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
        result = mixin._normalize_keywords_for_dedup(["абв"])
        assert isinstance(result, frozenset)

    def test_empty_list_returns_empty_frozenset(self, mixin):
        assert mixin._normalize_keywords_for_dedup([]) == frozenset()

    def test_lowercase(self, mixin):
        assert mixin._normalize_keywords_for_dedup(["АБВ"]) == frozenset({"абв"})

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
        """Слова короче 2 символов отбрасываются — это контракт нормализатора."""
        result = mixin._normalize_keywords_for_dedup(["a", "аб", "c"])
        assert "a" not in result
        assert "c" not in result
        assert "аб" in result

    def test_duplicates_removed(self, mixin):
        result = mixin._normalize_keywords_for_dedup(["АИС", "аис", "ais"])
        assert result == frozenset({"аис", "ais"})

    def test_non_string_values_handled(self, mixin):
        """str(kw) применяется ко всему — не должно падать."""
        result = mixin._normalize_keywords_for_dedup([12345, None])
        assert isinstance(result, frozenset)
        # "12345" (len 5) и "none" (len 4) — оба длиннее 2, попадают в набор
        assert "12345" in result
        assert "none" in result


class TestIsKeywordSetDuplicate:
    def test_empty_previous_returns_false(self, mixin):
        assert not mixin._is_keyword_set_duplicate(["аис", "система"], {"used_keyword_sets": set()})

    def test_same_set_detected(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["аис", "система"])
        state = {"used_keyword_sets": {norm}}
        assert mixin._is_keyword_set_duplicate(["аис", "система"], state)

    def test_case_insensitive_match(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["АИС"])
        state = {"used_keyword_sets": {norm}}
        assert mixin._is_keyword_set_duplicate(["аис"], state)

    def test_different_set_not_detected(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["аис", "система"])
        state = {"used_keyword_sets": {norm}}
        assert not mixin._is_keyword_set_duplicate(["доклад", "отчёт"], state)

    def test_subset_detected(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["аис", "система", "руководство"])
        state = {"used_keyword_sets": {norm}}
        assert mixin._is_keyword_set_duplicate(["аис", "система"], state)

    def test_superset_detected(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["аис"])
        state = {"used_keyword_sets": {norm}}
        assert mixin._is_keyword_set_duplicate(["аис", "система", "руководство"], state)

    def test_empty_keywords_returns_false(self, mixin):
        norm = mixin._normalize_keywords_for_dedup(["аис"])
        state = {"used_keyword_sets": {norm}}
        assert not mixin._is_keyword_set_duplicate([], state)

    def test_missing_used_keyword_sets_key(self, mixin):
        assert not mixin._is_keyword_set_duplicate(["аис"], {})

    def test_phrase_matches_words(self, mixin):
        """Фраза 'руководство аис' должна совпасть с ['аис', 'руководство']."""
        norm = mixin._normalize_keywords_for_dedup(["аис", "руководство"])
        state = {"used_keyword_sets": {norm}}
        assert mixin._is_keyword_set_duplicate(["руководство аис"], state)

    def test_all_short_words_normalize_to_empty(self, mixin):
        """Если все слова короче 2 символов — набор пуст, дубликат не срабатывает."""
        norm = mixin._normalize_keywords_for_dedup(["а"])
        assert norm == frozenset()
        state = {"used_keyword_sets": {norm}}
        assert not mixin._is_keyword_set_duplicate(["б", "в"], state)
