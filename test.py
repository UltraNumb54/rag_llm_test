
services/main_app/app/services/agent_service/context_expansion.py
import asyncio
from typing import Dict, List

from loguru import logger

from app.config import settings


class ContextExpansionMixin:
    def _count_total_subchunks(self, state: Dict) -> int:
        """
        Реальное число подчанков в accumulated_chunks с учётом merged-групп.
        Обычный чанк = 1, merged-группа = len(metadata['merged_chunk_ids']).
        Используется для контроля лимита AGENT_NEIGHBORS_MAX_TOTAL_CHUNKS.
        """
        total = 0
        for chunk in state["accumulated_chunks"].values():
            meta = chunk.get("metadata", {}) or {}
            if meta.get("is_merged"):
                total += len(meta.get("merged_chunk_ids", []) or [1])
            else:
                total += 1
        return total

    async def _get_document_context_batch(
        self,
        fragment_indices: List[int],
        qdrant_client,
        state: Dict,
    ) -> List[Dict]:
        """Пакетная загрузка соседних чанков для нескольких фрагментов"""
        accumulated_subchunks = self._count_total_subchunks(state)
        neighbors_max = settings.AGENT_NEIGHBORS_MAX_TOTAL_CHUNKS
        if accumulated_subchunks >= neighbors_max:
            logger.debug(
                f"get_document_context пропущен: "
                f"accumulated_subchunks={accumulated_subchunks} "
                f">= neighbors_max {neighbors_max}"
            )
            return []

        # Собираем запросы: на каждый документ — список chunk_id и ссылок на родительские чанки
        doc_requests: Dict[str, Dict] = {}
        for fragment_index in fragment_indices:
            ref_chunk = state["accumulated_chunks"].get(fragment_index)
            if not ref_chunk:
                continue
            display_name = ref_chunk["metadata"].get("display_name", "unknown")
            chunk_id = (
                ref_chunk["metadata"].get("chunk_id_numeric")
                or ref_chunk["metadata"].get("chunk_id", 0)
            )
            if display_name not in doc_requests:
                doc_requests[display_name] = {"chunk_ids": [], "refs": []}
            if chunk_id not in doc_requests[display_name]["chunk_ids"]:
                doc_requests[display_name]["chunk_ids"].append(chunk_id)
            doc_requests[display_name]["refs"].append((fragment_index, ref_chunk))

        if not doc_requests:
            return []

        logger.debug(
            f"get_document_context: {len(fragment_indices)} фрагментов -> "
            f"{len(doc_requests)} документов: "
            f"{[(dn, r['chunk_ids']) for dn, r in doc_requests.items()]}"
        )

        tasks = []
        doc_names = []
        for display_name, req in doc_requests.items():
            doc_names.append(display_name)
            tasks.append(
                qdrant_client.get_document_chunks_batch(
                    display_name=display_name,
                    chunk_ids=req["chunk_ids"],
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Группируем полученные чанки по документу и chunk_id
        doc_groups: Dict[str, Dict[int, Dict]] = {}
        for i, display_name in enumerate(doc_names):
            result = results[i]
            if isinstance(result, Exception):
                logger.warning(f"Ошибка чанков для '{display_name}': {result}")
                continue
            if not result or "chunks" not in result:
                continue
            chunks_data = result.get("chunks", [])
            if not chunks_data:
                continue

            refs = doc_requests[display_name]["refs"]
            # Берём МАКСИМАЛЬНЫЙ score среди всех родительских фрагментов,
            # ссылающихся на этот документ. Если на документ ссылаются
            # несколько выбранных чанков с разными score — берём лучший.
            parent_score = max(
                (r.get("score", 0.5) for _, r in refs),
                default=0.5,
            )

            for c in chunks_data:
                text = c.get("text", "")
                text_hash = hash(text.strip())
                if text_hash in state["seen_texts"]:
                    continue
                state["seen_texts"].add(text_hash)

                metadata = c.get("metadata", {})
                dn = metadata.get("display_name", display_name)
                cid = (
                    metadata.get("chunk_id_numeric")
                    or metadata.get("chunk_id", 0)
                )
                if dn not in doc_groups:
                    doc_groups[dn] = {}
                if cid in doc_groups[dn]:
                    continue
                doc_groups[dn][cid] = {
                    "text": text,
                    "metadata": metadata,
                    "score": parent_score,   # score родителя для всех подчанков
                }

        if not doc_groups:
            return []

        all_new_chunks: List[Dict] = []
        # Лимит по подчанкам: сколько ещё подчанков можно добавить.
        # Каждый новый чанк или merged-группа учитываются по числу подчанков внутри.
        remaining_slots = max(0, neighbors_max - accumulated_subchunks)

        for display_name, chunks_by_id in doc_groups.items():
            if remaining_slots <= 0:
                break
            sorted_ids = sorted(chunks_by_id.keys())

            if len(sorted_ids) == 1:
                # Одиночный сосед — занимает 1 слот
                if remaining_slots < 1:
                    break
                cid = sorted_ids[0]
                info = chunks_by_id[cid]
                doc_text = info["text"]
                metadata = info["metadata"]
                idx = state["next_index"]
                state["next_index"] += 1
                state["accumulated_chunks"][idx] = {
                    "document": doc_text,
                    "metadata": metadata,
                    "score": info["score"],   # = parent_score
                    "source": "neighbors",
                }
                all_new_chunks.append(
                    {
                        "index": idx,
                        "display_name": display_name,
                        "chunk_id": cid,
                        "relevance": round(info["score"], 4),
                        "text": self._shorten_text(
                            doc_text, settings.AGENT_CHUNK_TEXT_LIMIT
                        ),
                    }
                )
                remaining_slots -= 1
            else:
                # Merged-группа занимает len(sorted_ids) слотов.
                # Если не хватает слотов на всю группу — обрезаем её по
                # количеству оставшихся слотов, чтобы не переполнять лимит.
                if remaining_slots < 1:
                    break
                if len(sorted_ids) > remaining_slots:
                    logger.debug(
                        f"neighbors merged-группа для '{display_name}' "
                        f"обрезана с {len(sorted_ids)} до {remaining_slots} "
                        f"(лимит подчанков)"
                    )
                    sorted_ids = sorted_ids[:remaining_slots]

                document_parts: List[str] = []
                display_parts: List[str] = []
                merged_metadata = {}
                chunk_id_list = []
                # Score для всей merged-группы = score родительского чанка.
                parent_score = chunks_by_id[sorted_ids[0]]["score"]
                for pos, cid in enumerate(sorted_ids):
                    info = chunks_by_id[cid]
                    chunk_id_list.append(cid)
                    if not merged_metadata:
                        merged_metadata = dict(info["metadata"])
                    if pos > 0:
                        prev_id = sorted_ids[pos - 1]
                        if cid - prev_id > 1:
                            gap_start = prev_id + 1
                            gap_end = cid - 1
                            if gap_start == gap_end:
                                display_parts.append(
                                    f"[... чанк {gap_start} не загружен ...]"
                                )
                            else:
                                display_parts.append(
                                    f"[... чанки {gap_start}–{gap_end} "
                                    f"не загружены ...]"
                                )
                    full_text = info["text"]
                    document_parts.append(full_text)
                    shortened = self._shorten_text(
                        full_text, settings.AGENT_CHUNK_TEXT_LIMIT
                    )
                    display_parts.append(f"[chunk {cid}] {shortened}")

                merged_document_text = "\n".join(document_parts)
                merged_display_text = "\n".join(display_parts)
                merged_metadata["merged_chunk_ids"] = chunk_id_list
                merged_metadata["is_merged"] = True

                idx = state["next_index"]
                state["next_index"] += 1
                state["accumulated_chunks"][idx] = {
                    "document": merged_document_text,
                    "metadata": merged_metadata,
                    "score": parent_score,   # = score родителя для всей группы
                    "source": "neighbors",
                }
                ids_str = ",".join(str(c) for c in chunk_id_list)
                all_new_chunks.append(
                    {
                        "index": idx,
                        "display_name": display_name,
                        "chunk_id": ids_str,
                        "relevance": round(parent_score, 4),
                        "text": merged_display_text,
                    }
                )
                remaining_slots -= len(chunk_id_list)

        added_names = [
            f"[{c['index']}]{c['display_name']}" for c in all_new_chunks
        ]
        logger.debug(
            f"Для {fragment_indices} | "
            f"добавлено={len(all_new_chunks)} | "
            f"{', '.join(added_names) if added_names else '—'}"
        )
        return all_new_chunks

tests/unit/agent/test_context_expansion.py
"""
Unit-тесты для ContextExpansionMixin._get_document_context_batch и
_count_total_subchunks.

Проверяют:
- все подчанки получают score = max из scores родительских фрагментов;
- merged-группа тоже получает единый score родителя;
- дедупликация по seen_texts;
- раннюю остановку при превышении AGENT_NEIGHBORS_MAX_TOTAL_CHUNKS;
- обрезку merged-группы до оставшихся слотов (учёт по подчанкам);
- подсчёт _count_total_subchunks для обычных и merged-чанков.
"""
from unittest.mock import AsyncMock, patch
import pytest
from app.services.agent_service.context_expansion import ContextExpansionMixin


class TestableCtx(ContextExpansionMixin):
    def _shorten_text(self, text, limit):
        return text[:limit] if text else ""


@pytest.fixture
def mixin():
    return TestableCtx()


def _chunk(idx, doc, score, name="doc"):
    return {
        "document": doc,
        "metadata": {"display_name": name, "chunk_id": idx},
        "score": score, "source": "semantic",
    }


# ─── _count_total_subchunks ───

class TestCountTotalSubchunks:
    def test_empty(self, mixin):
        assert mixin._count_total_subchunks({"accumulated_chunks": {}}) == 0

    def test_single_chunks(self, mixin):
        state = {"accumulated_chunks": {
            1: _chunk(1, "a", 0.5),
            2: _chunk(2, "b", 0.5),
        }}
        assert mixin._count_total_subchunks(state) == 2

    def test_merged_group_counted_by_subchunks(self, mixin):
        state = {"accumulated_chunks": {
            1: _chunk(1, "a", 0.5),
            2: {
                "document": "merged",
                "metadata": {"is_merged": True, "merged_chunk_ids": [10, 11, 12, 13]},
                "score": 0.7, "source": "neighbors",
            },
        }}
        # 1 обычный + 4 подчанка = 5
        assert mixin._count_total_subchunks(state) == 5


# ─── _get_document_context_batch ───

class TestGetDocumentContextBatch:
    @pytest.mark.asyncio
    @patch("app.services.agent_service.context_expansion.settings")
    async def test_returns_empty_when_limit_reached(self, mock_settings, mixin):
        mock_settings.AGENT_NEIGHBORS_MAX_TOTAL_CHUNKS = 1
        state = {
            "accumulated_chunks": {1: _chunk(1, "родитель", 0.7, "docA")},
            "seen_texts": set(), "next_index": 2,
        }
        qdrant = AsyncMock()
        result = await mixin._get_document_context_batch([1], qdrant, state)
        assert result == []
        qdrant.get_document_chunks_batch.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.agent_service.context_expansion.settings")
    async def test_single_neighbor_gets_parent_score(self, mock_settings, mixin):
        mock_settings.AGENT_NEIGHBORS_MAX_TOTAL_CHUNKS = 32
        mock_settings.AGENT_CHUNK_TEXT_LIMIT = 1000
        state = {
            "accumulated_chunks": {1: _chunk(1, "родитель", 0.85, "docA")},
            "seen_texts": set(), "next_index": 2,
        }
        qdrant = AsyncMock()
        qdrant.get_document_chunks_batch = AsyncMock(return_value={
            "chunks": [
                {"text": "сосед", "metadata": {"display_name": "docA", "chunk_id": 2}},
            ]
        })
        result = await mixin._get_document_context_batch([1], qdrant, state)
        assert len(result) == 1
        assert result[0]["relevance"] == 0.85
        assert state["accumulated_chunks"][2]["score"] == 0.85

    @pytest.mark.asyncio
    @patch("app.services.agent_service.context_expansion.settings")
    async def test_max_score_wins_across_refs(self, mock_settings, mixin):
        """Если на один документ ссылаются несколько фрагментов — берём max score."""
        mock_settings.AGENT_NEIGHBORS_MAX_TOTAL_CHUNKS = 32
        mock_settings.AGENT_CHUNK_TEXT_LIMIT = 1000
        state = {
            "accumulated_chunks": {
                1: _chunk(1, "родитель 1", 0.3, "docA"),
                2: _chunk(2, "родитель 2", 0.9, "docA"),
            },
            "seen_texts": set(), "next_index": 3,
        }
        qdrant = AsyncMock()
        qdrant.get_document_chunks_batch = AsyncMock(return_value={
            "chunks": [{"text": "сосед", "metadata": {"display_name": "docA", "chunk_id": 99}}]
        })
        result = await mixin._get_document_context_batch([1, 2], qdrant, state)
        assert result[0]["relevance"] == 0.9

    @pytest.mark.asyncio
    @patch("app.services.agent_service.context_expansion.settings")
    async def test_merged_group_uses_parent_score(self, mock_settings, mixin):
        mock_settings.AGENT_NEIGHBORS_MAX_TOTAL_CHUNKS = 32
        mock_settings.AGENT_CHUNK_TEXT_LIMIT = 1000
        state = {
            "accumulated_chunks": {1: _chunk(1, "родитель", 0.6, "docA")},
            "seen_texts": set(), "next_index": 2,
        }
        qdrant = AsyncMock()
        qdrant.get_document_chunks_batch = AsyncMock(return_value={
            "chunks": [
                {"text": "c1", "metadata": {"display_name": "docA", "chunk_id": 2}},
                {"text": "c2", "metadata": {"display_name": "docA", "chunk_id": 3}},
                {"text": "c3", "metadata": {"display_name": "docA", "chunk_id": 4}},
            ]
        })
        result = await mixin._get_document_context_batch([1], qdrant, state)
        assert len(result) == 1
        assert result[0]["relevance"] == 0.6
        # merged-группа в accumulated_chunks получает тот же score
        merged_idx = result[0]["index"]
        assert state["accumulated_chunks"][merged_idx]["score"] == 0.6
        assert state["accumulated_chunks"][merged_idx]["metadata"]["is_merged"] is True
        assert state["accumulated_chunks"][merged_idx]["metadata"]["merged_chunk_ids"] == [2, 3, 4]

    @pytest.mark.asyncio
    @patch("app.services.agent_service.context_expansion.settings")
    async def test_merged_group_truncated_to_remaining_slots(self, mock_settings, mixin):
        """Merged-группа обрезается до оставшихся слотов, чтобы не превышать лимит."""
        mock_settings.AGENT_NEIGHBORS_MAX_TOTAL_CHUNKS = 3
        mock_settings.AGENT_CHUNK_TEXT_LIMIT = 1000
        state = {
            # уже 1 подчанок -> осталось 2 слота
            "accumulated_chunks": {1: _chunk(1, "родитель", 0.5, "docA")},
            "seen_texts": set(), "next_index": 2,
        }
        qdrant = AsyncMock()
        qdrant.get_document_chunks_batch = AsyncMock(return_value={
            "chunks": [
                {"text": f"c{i}", "metadata": {"display_name": "docA", "chunk_id": i}}
                for i in range(2, 8)  # 6 подчанков
            ]
        })
        result = await mixin._get_document_context_batch([1], qdrant, state)
        assert len(result) == 1
        merged = state["accumulated_chunks"][result[0]["index"]]
        # Ровно 2 подчанка (не 6), чтобы не превысить лимит 3
        assert len(merged["metadata"]["merged_chunk_ids"]) == 2
        assert mixin._count_total_subchunks(state) == 3

    @pytest.mark.asyncio
    @patch("app.services.agent_service.context_expansion.settings")
    async def test_deduplication_by_seen_texts(self, mock_settings, mixin):
        mock_settings.AGENT_NEIGHBORS_MAX_TOTAL_CHUNKS = 32
        mock_settings.AGENT_CHUNK_TEXT_LIMIT = 1000
        state = {
            "accumulated_chunks": {1: _chunk(1, "родитель", 0.5, "docA")},
            "seen_texts": {hash("уже_видели")},
            "next_index": 2,
        }
        qdrant = AsyncMock()
        qdrant.get_document_chunks_batch = AsyncMock(return_value={
            "chunks": [
                {"text": "уже_видели", "metadata": {"display_name": "docA", "chunk_id": 2}},
                {"text": "новый", "metadata": {"display_name": "docA", "chunk_id": 3}},
            ]
        })
        result = await mixin._get_document_context_batch([1], qdrant, state)
        # "уже_видели" отфильтрован, остаётся только "новый" как одиночный чанк
        assert len(result) == 1
        assert result[0]["chunk_id"] == 3
