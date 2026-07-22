from __future__ import annotations

import operator
import re
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np


class Retriever:
    UPSTREAM_COMMIT = "df92a34d3ad824fe4ef45b81fc956498a353a943"
    SERIES_KEYS = (
        "Series Subject",
        "series_subject",
        "series",
        "Series",
        "series_number",
    )
    ID_KEYS = (
        "doc_id",
        "document_id",
        "Document ID",
        "id",
        "ID",
        "source",
        "Source",
    )
    GLOSSARY = {
        "5gc": "5G Core",
        "amf": "Access and Mobility Management Function",
        "ausf": "Authentication Server Function",
        "enb": "evolved Node B",
        "gnb": "next generation Node B",
        "ims": "IP Multimedia Subsystem",
        "mme": "Mobility Management Entity",
        "nas": "Non-Access Stratum",
        "nrf": "Network Repository Function",
        "nssf": "Network Slice Selection Function",
        "pcf": "Policy Control Function",
        "pdu": "Protocol Data Unit",
        "qos": "Quality of Service",
        "ran": "Radio Access Network",
        "rrc": "Radio Resource Control",
        "smf": "Session Management Function",
        "udm": "Unified Data Management",
        "ue": "User Equipment",
        "upf": "User Plane Function",
    }

    def __init__(
        self,
        corpus_rows: Sequence[Mapping[str, Any]],
        document_embeddings: Any,
        embed_query: Callable[[str], Any],
        cache_dir: Any = None,
    ) -> None:
        if hasattr(corpus_rows, "to_dict"):
            corpus_rows = corpus_rows.to_dict("records")
        self.rows = list(corpus_rows)
        if not self.rows:
            raise ValueError("corpus_rows must not be empty")
        if any(not isinstance(row, Mapping) for row in self.rows):
            raise TypeError("each corpus row must be a mapping")
        if not callable(embed_query):
            raise TypeError("embed_query must be callable")

        embeddings = np.asarray(document_embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("document_embeddings must be a 2-D array")
        if embeddings.shape[0] != len(self.rows):
            raise ValueError("corpus_rows and document_embeddings must align")
        if embeddings.shape[1] == 0:
            raise ValueError("document_embeddings must have a nonzero dimension")
        if not np.isfinite(embeddings).all():
            raise ValueError("document_embeddings contain non-finite values")

        self.embed_query = embed_query
        self.dimension = int(embeddings.shape[1])
        self.document_embeddings = self._normalize_rows(embeddings)
        self.series = [self._series_number(row) for row in self.rows]
        self.doc_ids = [
            str(self._row_value(row, self.ID_KEYS, index))
            for index, row in enumerate(self.rows)
        ]
        self.centroids = self._build_centroids()

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(
            values,
            norms,
            out=np.zeros_like(values, dtype=np.float32),
            where=norms > 0,
        )

    @staticmethod
    def _normalize_vector(value: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        if norm == 0.0:
            return np.zeros_like(value, dtype=np.float32)
        return value / norm

    @staticmethod
    def _containers(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        containers = [row]
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            containers.append(metadata)
        return containers

    @classmethod
    def _row_value(
        cls,
        row: Mapping[str, Any],
        keys: Sequence[str],
        default: Any,
    ) -> Any:
        for container in cls._containers(row):
            for key in keys:
                value = container.get(key)
                if value is not None and str(value).strip():
                    return value
        return default

    @classmethod
    def _series_number(cls, row: Mapping[str, Any]) -> str | None:
        value = cls._row_value(row, cls.SERIES_KEYS, None)
        if value is None:
            return None
        if isinstance(value, (int, np.integer)) and 21 <= int(value) <= 38:
            return str(int(value))
        if isinstance(value, float) and value.is_integer() and 21 <= int(value) <= 38:
            return str(int(value))
        match = re.search(r"(?<!\d)(2[1-9]|3[0-8])(?!\d)", str(value))
        return match.group(1) if match else None

    def _build_centroids(self) -> dict[str, np.ndarray]:
        centroids: dict[str, np.ndarray] = {}
        labels = sorted({item for item in self.series if item}, key=int)
        series_array = np.asarray(self.series, dtype=object)
        for label in labels:
            indices = np.flatnonzero(series_array == label)
            centroid = self.document_embeddings[indices].mean(axis=0)
            centroids[label] = self._normalize_vector(centroid)
        return centroids

    @classmethod
    def _expand_query(cls, question: str) -> tuple[str, list[str]]:
        definitions = []
        for acronym, long_form in cls.GLOSSARY.items():
            pattern = rf"(?<![A-Za-z0-9]){re.escape(acronym)}(?![A-Za-z0-9])"
            if re.search(pattern, question, flags=re.IGNORECASE):
                if long_form.lower() not in question.lower():
                    definitions.append(f"{acronym.upper()} means {long_form}")
        if not definitions:
            return question.strip(), []
        expanded = question.strip() + "\n3GPP terms: " + "; ".join(definitions)
        return expanded, definitions

    def _embed(self, text: str) -> np.ndarray:
        vector = np.asarray(self.embed_query(text), dtype=np.float32).reshape(-1)
        if vector.size != self.dimension:
            raise ValueError(
                "query embedding dimension does not match document embeddings"
            )
        if not np.isfinite(vector).all():
            raise ValueError("query embedding contains non-finite values")
        return self._normalize_vector(vector)

    def _route(self, query_vector: np.ndarray) -> tuple[list[str], list[dict[str, Any]]]:
        if not self.centroids or not np.any(query_vector):
            return [], []
        scored = [
            (label, float(np.dot(query_vector, centroid)))
            for label, centroid in self.centroids.items()
        ]
        scored.sort(key=lambda item: (-item[1], int(item[0])))
        selected = scored[: min(3, len(scored))]
        return (
            [label for label, _ in selected],
            [{"series": label, "score": score} for label, score in selected],
        )

    def _search_pass(self, query_text: str, top_k: int) -> dict[str, Any]:
        query_vector = self._embed(query_text)
        selected_series, router_scores = self._route(query_vector)
        if selected_series:
            candidate_indices = np.asarray(
                [
                    index
                    for index, label in enumerate(self.series)
                    if label in selected_series
                ],
                dtype=np.int64,
            )
        else:
            candidate_indices = np.arange(len(self.rows), dtype=np.int64)

        if candidate_indices.size == 0:
            candidate_indices = np.arange(len(self.rows), dtype=np.int64)
        scores = self.document_embeddings[candidate_indices] @ query_vector
        order = np.lexsort((candidate_indices, -scores))[:top_k]
        result_indices = candidate_indices[order]
        result_scores = scores[order]
        return {
            "query": query_text,
            "selected_series": selected_series,
            "router_scores": router_scores,
            "candidate_document_count": int(candidate_indices.size),
            "indices": result_indices.tolist(),
            "doc_ids": [self.doc_ids[index] for index in result_indices],
            "scores": [float(score) for score in result_scores],
        }

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        candidate_answer: str = "",
    ) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        try:
            requested_k = operator.index(top_k)
        except TypeError as exc:
            raise TypeError("top_k must be an integer") from exc
        if requested_k <= 0:
            raise ValueError("top_k must be positive")
        requested_k = min(requested_k, len(self.rows))

        expanded_question, definitions = self._expand_query(question)
        first_pass = self._search_pass(expanded_question, requested_k)
        passes = [first_pass]

        if isinstance(candidate_answer, str) and candidate_answer.strip():
            second_query = (
                expanded_question
                + "\nCandidate answer for retrieval refinement: "
                + candidate_answer.strip()
            )
            passes.append(self._search_pass(second_query, requested_k))

        final_pass = passes[-1]
        indices = final_pass["indices"]
        trace_passes = []
        for number, item in enumerate(passes, start=1):
            trace_passes.append(
                {
                    "pass": number,
                    "query": item["query"],
                    "selected_series": item["selected_series"],
                    "router_scores": item["router_scores"],
                    "candidate_document_count": item["candidate_document_count"],
                    "doc_ids": item["doc_ids"],
                    "scores": item["scores"],
                }
            )

        return {
            "contexts": [dict(self.rows[index]) for index in indices],
            "doc_ids": list(final_pass["doc_ids"]),
            "scores": list(final_pass["scores"]),
            "trace": {
                "adapter": "telco-rag-common-model-adapter",
                "upstream_commit": self.UPSTREAM_COMMIT,
                "query_expansion": {
                    "expanded_question": expanded_question,
                    "definitions": definitions,
                },
                "router": "top-3 series centroids by cosine similarity",
                "retrieval": "dense cosine search within routed series",
                "pass_count": len(trace_passes),
                "passes": trace_passes,
                "candidate_answer_supplied": len(trace_passes) == 2,
            },
        }
