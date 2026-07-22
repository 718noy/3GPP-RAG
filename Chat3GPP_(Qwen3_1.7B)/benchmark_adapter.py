from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TEXT_FIELDS = (
    "Section",
    "Content",
    "Source",
    "Document Title",
    "Working Group",
    "Series Subject",
    "Subclause",
)


class Retriever:
    BM25_K1 = 1.5
    BM25_B = 0.75
    RRF_K = 10
    FIRST_STAGE_LIMIT = 1000
    RERANK_LIMIT = 50
    COSINE_WEIGHT = 0.7
    RRF_WEIGHT = 0.3

    def __init__(
        self,
        corpus_rows: Sequence[Mapping[str, Any]],
        document_embeddings: Any,
        embed_query: Callable[[str], Any],
        cache_dir: str | None = None,
    ) -> None:
        if not callable(embed_query):
            raise TypeError("embed_query must be callable")

        self.corpus_rows = [dict(row) for row in corpus_rows]
        self.embed_query = embed_query
        missing_ids = [i for i, row in enumerate(self.corpus_rows) if "doc_id" not in row]
        if missing_ids:
            raise ValueError(f"corpus row {missing_ids[0]} has no doc_id")

        embeddings = np.asarray(document_embeddings, dtype=np.float32)
        if embeddings.ndim == 1 and len(self.corpus_rows) == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim != 2:
            raise ValueError("document_embeddings must be a two-dimensional array")
        if embeddings.shape[0] != len(self.corpus_rows):
            raise ValueError("document_embeddings row count must match corpus_rows")
        if embeddings.shape[1] == 0:
            raise ValueError("document_embeddings must have at least one dimension")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._document_embeddings = np.divide(
            embeddings,
            norms,
            out=np.zeros_like(embeddings),
            where=norms > 0,
        )

        self._token_counts: list[Counter[str]] = []
        self._document_lengths = np.zeros(len(self.corpus_rows), dtype=np.float32)
        document_frequency: Counter[str] = Counter()
        for index, row in enumerate(self.corpus_rows):
            tokens = self._tokenize(self._row_text(row))
            counts = Counter(tokens)
            self._token_counts.append(counts)
            self._document_lengths[index] = len(tokens)
            document_frequency.update(counts.keys())

        count = len(self.corpus_rows)
        self._average_document_length = (
            float(np.mean(self._document_lengths)) if count else 0.0
        )
        self._inverse_document_frequency = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    @staticmethod
    def _row_text(row: Mapping[str, Any]) -> str:
        values = []
        for field in _TEXT_FIELDS:
            value = row.get(field, "")
            if value is not None:
                values.append(str(value))
        return " ".join(values)

    def _bm25_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus_rows), dtype=np.float32)
        if not query_tokens or not self.corpus_rows:
            return scores

        average_length = self._average_document_length or 1.0
        for term in set(query_tokens):
            idf = self._inverse_document_frequency.get(term)
            if idf is None:
                continue
            for index, counts in enumerate(self._token_counts):
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                length_ratio = self._document_lengths[index] / average_length
                denominator = frequency + self.BM25_K1 * (
                    1.0 - self.BM25_B + self.BM25_B * length_ratio
                )
                scores[index] += idf * (
                    frequency * (self.BM25_K1 + 1.0) / denominator
                )
        return scores

    def _query_vector(self, question: str) -> np.ndarray:
        vector = np.asarray(self.embed_query(question), dtype=np.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1:
            raise ValueError("embed_query must return one vector")
        if vector.shape[0] != self._document_embeddings.shape[1]:
            raise ValueError("query and document embedding dimensions do not match")
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else np.zeros_like(vector)

    @staticmethod
    def _rank(scores: np.ndarray, limit: int) -> list[int]:
        if not len(scores) or limit <= 0:
            return []
        return np.argsort(-scores, kind="stable")[:limit].tolist()

    def retrieve(self, question: str, top_k: int = 5) -> dict[str, Any]:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self.corpus_rows:
            return {
                "contexts": [],
                "doc_ids": [],
                "scores": [],
                "trace": self._trace_header(0, 0, []),
            }

        query_tokens = self._tokenize(question)
        query_vector = self._query_vector(question)
        bm25_scores = self._bm25_scores(query_tokens)
        cosine_scores = self._document_embeddings @ query_vector

        first_stage_limit = min(self.FIRST_STAGE_LIMIT, len(self.corpus_rows))
        bm25_order = self._rank(bm25_scores, first_stage_limit)
        dense_order = self._rank(cosine_scores, first_stage_limit)
        bm25_ranks = {index: rank for rank, index in enumerate(bm25_order, start=1)}
        dense_ranks = {index: rank for rank, index in enumerate(dense_order, start=1)}

        rrf_scores: dict[int, float] = {}
        for index, rank in bm25_ranks.items():
            rrf_scores[index] = rrf_scores.get(index, 0.0) + 1.0 / (self.RRF_K + rank)
        for index, rank in dense_ranks.items():
            rrf_scores[index] = rrf_scores.get(index, 0.0) + 1.0 / (self.RRF_K + rank)

        fused_order = sorted(rrf_scores, key=lambda i: (-rrf_scores[i], i))
        rerank_candidates = fused_order[: min(self.RERANK_LIMIT, len(fused_order))]
        max_rrf = max((rrf_scores[i] for i in rerank_candidates), default=1.0)

        final_scores: dict[int, float] = {}
        for index in rerank_candidates:
            cosine_unit = max(0.0, min(1.0, (float(cosine_scores[index]) + 1.0) / 2.0))
            rrf_unit = rrf_scores[index] / max_rrf if max_rrf > 0 else 0.0
            final_scores[index] = (
                self.COSINE_WEIGHT * cosine_unit + self.RRF_WEIGHT * rrf_unit
            )

        final_order = sorted(
            rerank_candidates,
            key=lambda i: (-final_scores[i], -rrf_scores[i], i),
        )[: min(top_k, len(rerank_candidates))]

        ranking_trace = []
        for index in final_order:
            row = self.corpus_rows[index]
            ranking_trace.append(
                {
                    "doc_id": str(row["doc_id"]),
                    "bm25_rank": bm25_ranks.get(index),
                    "dense_rank": dense_ranks.get(index),
                    "bm25_score": float(bm25_scores[index]),
                    "cosine_score": float(cosine_scores[index]),
                    "rrf_score": float(rrf_scores[index]),
                    "final_score": float(final_scores[index]),
                }
            )

        return {
            "contexts": [dict(self.corpus_rows[i]) for i in final_order],
            "doc_ids": [str(self.corpus_rows[i]["doc_id"]) for i in final_order],
            "scores": [float(final_scores[i]) for i in final_order],
            "trace": self._trace_header(
                len(fused_order), len(rerank_candidates), ranking_trace
            ),
        }

    def _trace_header(
        self,
        fused_candidate_count: int,
        rerank_candidate_count: int,
        ranking: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "system": "chat3gpp_common_model_adapter",
            "paper": "Chat3GPP (arXiv:2501.13954)",
            "method": "BM25 + dense cosine -> RRF -> common-embedding rerank",
            "first_stage_limit_per_retriever": min(
                self.FIRST_STAGE_LIMIT, len(self.corpus_rows)
            ),
            "rrf_k": self.RRF_K,
            "fused_candidate_count": fused_candidate_count,
            "rerank_candidate_count": rerank_candidate_count,
            "rerank_limit": self.RERANK_LIMIT,
            "cosine_weight": self.COSINE_WEIGHT,
            "rrf_weight": self.RRF_WEIGHT,
            "ranking": ranking,
            "adaptation_notice": (
                "The paper uses BGE-M3/ColBERT components. This benchmark adapter "
                "uses the shared BGE-small embeddings and cosine scoring, so it is "
                "a controlled reimplementation rather than an exact reproduction."
            ),
        }
