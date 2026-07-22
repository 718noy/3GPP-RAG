from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Iterable


for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(2_147_483_647 if sys.platform == "win32" else sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def document_text(row: dict[str, str]) -> str:
    heading = " | ".join(
        value for value in (row.get("Source", ""), row.get("Section", ""), row.get("Document Title", "")) if value
    )
    return f"{heading}\n{row.get('Content', '')}".strip()


class SharedEmbedder:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        corpus_path: Path,
        batch_size: int = 4,
        revision: str | None = None,
    ) -> None:
        if batch_size < 1 or batch_size > 4:
            raise ValueError("batch_size must stay between 1 and 4 for this low-load package")
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        self.np = __import__("numpy")
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            cache_folder=str(self.cache_dir / "models"),
            revision=revision,
        )
        self.corpus_hash = sha256_file(corpus_path)

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            device="cpu",
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 32,
            convert_to_numpy=True,
        ).astype("float32")

    def query(self, text: str):
        return self.encode([text])[0]

    def corpus_embeddings(self, rows: list[dict[str, str]]):
        matrix_path = self.cache_dir / "document_embeddings.npy"
        meta_path = self.cache_dir / "document_embeddings.json"
        expected = {
            "model": self.model_name,
            "revision": self.revision,
            "corpus_sha256": self.corpus_hash,
            "rows": len(rows),
        }
        if matrix_path.exists() and meta_path.exists():
            actual = json.loads(meta_path.read_text(encoding="utf-8"))
            if all(actual.get(key) == value for key, value in expected.items()):
                return self.np.load(matrix_path, mmap_mode="r")

        texts = [document_text(row) for row in rows]
        parts = []
        cursor = 0
        stages = ((1, 1, 5.0), (8, 2, 5.0))
        for count, stage_batch, pause in stages:
            if cursor >= len(texts):
                break
            end = min(len(texts), cursor + count)
            old_batch = self.batch_size
            self.batch_size = min(stage_batch, old_batch)
            parts.append(self.encode(texts[cursor:end]))
            self.batch_size = old_batch
            cursor = end
            time.sleep(pause)
        while cursor < len(texts):
            end = min(len(texts), cursor + 128)
            parts.append(self.encode(texts[cursor:end]))
            cursor = end
            time.sleep(0.5)
        matrix = self.np.concatenate(parts, axis=0)
        self.np.save(matrix_path, matrix)
        meta_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        return matrix


def gpu_snapshot() -> dict[str, int] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        if completed.returncode:
            return None
        values = [int(part.strip()) for part in completed.stdout.splitlines()[0].split(",")]
        return dict(zip(("temperature", "utilization", "memory_used", "memory_total"), values))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class ResourceGuard:
    def __init__(self, temperature_limit: int = 72, memory_limit_mb: int = 8192, cpu_limit: int = 65) -> None:
        self.temperature_limit = temperature_limit
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit = cpu_limit

    @staticmethod
    def cpu_percent() -> float | None:
        try:
            import psutil

            return float(psutil.cpu_percent(interval=1.0))
        except (ImportError, OSError):
            return None

    def wait_until_cool(self) -> None:
        while True:
            state = gpu_snapshot()
            if state is None:
                return
            hot = state["temperature"] >= self.temperature_limit
            crowded = state["memory_used"] >= self.memory_limit_mb
            busy = state["utilization"] >= 70
            cpu = self.cpu_percent()
            cpu_busy = cpu is not None and cpu >= self.cpu_limit
            if not (hot or crowded or busy or cpu_busy):
                return
            print(f"Resource guard waiting: gpu={state} cpu={cpu}", flush=True)
            time.sleep(10)


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3:1.7b",
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        cooldown_seconds: float = 3.0,
        num_gpu_layers: int = 2,
    ) -> None:
        import requests

        self.requests = requests
        self.model = model
        self.endpoint = endpoint
        self.cooldown_seconds = max(2.0, cooldown_seconds)
        self.num_gpu_layers = max(0, min(8, num_gpu_layers))
        self.guard = ResourceGuard()
        self.calls = 0

    def _chat(self, prompt: str, num_predict: int) -> tuple[str, float]:
        self.guard.wait_until_cool()
        if self.calls == 0:
            time.sleep(5)
        if self.calls == 0:
            active_gpu_layers = 0
        elif self.calls < 3:
            active_gpu_layers = min(1, self.num_gpu_layers)
        elif self.calls < 6:
            active_gpu_layers = min(2, self.num_gpu_layers)
        elif self.calls < 9:
            active_gpu_layers = min(4, self.num_gpu_layers)
        else:
            active_gpu_layers = self.num_gpu_layers
        started = time.perf_counter()
        response = self.requests.post(
            self.endpoint,
            json={
                "model": self.model,
                "stream": False,
                "think": False,
                "keep_alive": "2m",
                "messages": [{"role": "user", "content": "/no_think\n" + prompt}],
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": 2048,
                    "num_predict": num_predict,
                    "num_thread": 2,
                    "num_batch": 16,
                    "num_gpu": active_gpu_layers,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - started
        self.calls += 1
        pause = 8.0 if self.calls <= 2 else self.cooldown_seconds
        time.sleep(pause)
        self.guard.wait_until_cool()
        content = response.json().get("message", {}).get("content", "")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content, elapsed

    def candidate_answer(self, question: str, contexts: list[dict[str, str]]) -> tuple[str, float]:
        evidence = format_context(contexts, max_chars=1800, question=question)
        prompt = (
            "Use the evidence to propose one short plausible answer to the question. "
            "Do not discuss answer choices. Return one sentence only.\n\n"
            f"Evidence:\n{evidence}\n\nQuestion: {question}\nAnswer:"
        )
        return self._chat(prompt, num_predict=24)

    def answer_mcq(self, question: dict, contexts: list[dict[str, str]]) -> tuple[str, float]:
        evidence = format_context(contexts, max_chars=2500, question=question["question"])
        options = "\n".join(f"{letter}. {question['options'][letter]}" for letter in "ABCD")
        prompt = (
            "You answer 3GPP multiple-choice questions. Use only the supplied evidence. "
            "Return exactly one capital letter: A, B, C, or D.\n\n"
            f"Evidence:\n{evidence}\n\nQuestion: {question['question']}\n{options}\nAnswer:"
        )
        return self._chat(prompt, num_predict=4)


def _relevant_excerpt(content: str, question: str, budget: int) -> str:
    if len(content) <= budget:
        return content
    terms = {term.lower() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", question) if len(term) > 2}
    step = max(80, budget // 2)
    best = content[:budget]
    best_score = -1
    for start in range(0, max(1, len(content) - budget + 1), step):
        window = content[start:start + budget]
        lowered = window.lower()
        score = sum(1 for term in terms if term in lowered)
        if score > best_score:
            best, best_score = window, score
    return best.strip()


def format_context(contexts: Iterable[dict[str, str]], max_chars: int, question: str) -> str:
    context_list = list(contexts)
    if not context_list:
        return ""
    per_document = max(240, max_chars // len(context_list) - 100)
    blocks = []
    used = 0
    for index, row in enumerate(context_list, start=1):
        excerpt = _relevant_excerpt(row.get("Content", ""), question, per_document)
        block = f"[{index}] {row.get('Source', '')} {row.get('Section', '')}\n{excerpt}".strip()
        remaining = max_chars - used
        if remaining <= 0:
            break
        block = block[:remaining]
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def parse_choice(text: str) -> str | None:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().upper()
    match = re.search(r"(?:^|\b)([A-D])(?:\b|$)", cleaned)
    return match.group(1) if match else None


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return center - spread, center + spread
