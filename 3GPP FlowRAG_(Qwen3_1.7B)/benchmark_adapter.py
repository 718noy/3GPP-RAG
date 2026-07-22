from __future__ import annotations

from collections import defaultdict
import re

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
PROCEDURE_WORDS = {
    "after", "before", "failure", "fails", "if", "message", "next", "procedure",
    "reject", "request", "response", "retry", "state", "success", "timer", "when",
}


def tokens(text: str) -> set[str]:
    return {value.lower() for value in TOKEN_RE.findall(text)}


def section_key(value: str):
    pieces = re.findall(r"\d+|[A-Za-z]+", value)
    return tuple((0, int(piece)) if piece.isdigit() else (1, piece.lower()) for piece in pieces)


def query_frame(question: str) -> dict:
    query_tokens = tokens(question)
    if {"before", "prior", "preceding"} & query_tokens:
        direction = "backward"
    elif {"after", "next", "following", "then"} & query_tokens:
        direction = "forward"
    else:
        direction = "both"
    conditions = query_tokens & PROCEDURE_WORDS
    return {"direction": direction, "conditions": sorted(conditions)}


class Retriever:
    def __init__(self, corpus_rows, document_embeddings, embed_query, cache_dir=None):
        self.rows = list(corpus_rows)
        self.embeddings = np.asarray(document_embeddings, dtype=np.float32)
        self.embed_query = embed_query
        self.token_sets = [tokens(f"{row.get('Section', '')} {row.get('Content', '')}") for row in self.rows]
        self.neighbors = self._build_neighbors()

    def _build_neighbors(self):
        groups = defaultdict(list)
        for index, row in enumerate(self.rows):
            groups[row.get("Source", "")].append(index)
        neighbors = {}
        for indices in groups.values():
            ordered = sorted(indices, key=lambda index: section_key(self.rows[index].get("Section", "")))
            for position, index in enumerate(ordered):
                neighbors[index] = (
                    ordered[position - 1] if position else None,
                    ordered[position + 1] if position + 1 < len(ordered) else None,
                )
        return neighbors

    def _path(self, seed: int, direction: str) -> list[int]:
        previous, following = self.neighbors.get(seed, (None, None))
        if direction == "forward":
            return [index for index in (seed, following) if index is not None]
        if direction == "backward":
            return [index for index in (previous, seed) if index is not None]
        return [index for index in (previous, seed, following) if index is not None]

    def retrieve(self, question: str, top_k: int = 5):
        query_vector = np.asarray(self.embed_query(question), dtype=np.float32)
        dense = self.embeddings @ query_vector
        candidate_count = min(80, len(self.rows))
        candidates = np.argpartition(-dense, candidate_count - 1)[:candidate_count]
        frame = query_frame(question)
        question_tokens = tokens(question)
        condition_tokens = set(frame["conditions"])
        ranked = []
        for seed in candidates:
            path = self._path(int(seed), frame["direction"])
            union = set().union(*(self.token_sets[index] for index in path))
            lexical = len(question_tokens & union) / max(1, len(question_tokens))
            condition = len(condition_tokens & union) / max(1, len(condition_tokens))
            procedure = len(PROCEDURE_WORDS & union) / len(PROCEDURE_WORDS)
            coherence = 1.0 if len(path) > 1 else 0.0
            semantic = (float(dense[int(seed)]) + 1.0) / 2.0
            score = 0.55 * semantic + 0.20 * lexical + 0.15 * condition + 0.05 * procedure + 0.05 * coherence
            ranked.append((score, int(seed), path))
        ranked.sort(reverse=True)

        selected = []
        scores = []
        seen = set()
        selected_paths = []
        for path_score, seed, path in ranked:
            path_ids = []
            for index in path:
                doc_id = str(self.rows[index].get("doc_id", index))
                path_ids.append(doc_id)
                if doc_id in seen or len(selected) >= top_k:
                    continue
                seen.add(doc_id)
                selected.append(self.rows[index])
                scores.append(path_score)
            selected_paths.append({"seed_doc_id": str(self.rows[seed].get("doc_id", seed)), "doc_ids": path_ids, "score": path_score})
            if len(selected) >= top_k:
                break
        return {
            "contexts": selected,
            "doc_ids": [str(row.get("doc_id", "")) for row in selected],
            "scores": scores,
            "trace": {"frame": frame, "paths": selected_paths[:3], "method": "condition_aware_ordered_section_paths"},
        }
