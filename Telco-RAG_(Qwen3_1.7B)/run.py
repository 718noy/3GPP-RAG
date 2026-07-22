from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import requests

from common_runtime import (
    OllamaClient,
    SharedEmbedder,
    load_corpus,
    load_questions,
    parse_choice,
    sha256_file,
    wilson_interval,
)


SYSTEM_ID = "telco_rag"
DISPLAY_NAME = "Telco-RAG-based RAG"
TWO_PASS_RETRIEVAL = True

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "Q&A set"
QA_PATH = DATA_ROOT / "qa_set.jsonl"
CORPUS_PATH = DATA_ROOT / "documents_TelcoBench.csv"
ADAPTER_PATH = ROOT / "benchmark_adapter.py"
CACHE_ROOT = DATA_ROOT / ".cache"
RESULTS_ROOT = ROOT / "results"

QA_SHA256 = "52a8b668380af0836f591546f6200fbb3fd9d3af9a2fb82a83a8238793de1049"
CORPUS_SHA256 = "ea7d6654e879e87738e0c88a8dcdeb29dcc58434815ea7e4a6ebdc64362effef"
CORPUS_URL = (
    "https://raw.githubusercontent.com/gagan-iitb/MM-TelcoBench/"
    "8f190784a01a6fad73851d09b09076d2617c1cda/"
    "Telecom_Information_Retrieval/documents_TelcoBench.csv"
)
DEFAULT_MODEL = "qwen3:1.7b"
MODEL_BLOB_SHA256 = "3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"
MODEL_MANIFEST_SHA256 = "8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7"


def ensure_inputs() -> None:
    if not QA_PATH.exists():
        raise FileNotFoundError(f"Missing Q&A file: {QA_PATH}")
    actual_qa = sha256_file(QA_PATH)
    if actual_qa != QA_SHA256:
        raise RuntimeError(f"Q&A hash mismatch: expected {QA_SHA256}, got {actual_qa}")

    if CORPUS_PATH.exists():
        actual_corpus = sha256_file(CORPUS_PATH)
        if actual_corpus != CORPUS_SHA256:
            raise RuntimeError(
                f"Corpus hash mismatch: expected {CORPUS_SHA256}, got {actual_corpus}"
            )
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = DATA_ROOT / "documents_TelcoBench.csv.partial"
    temporary.unlink(missing_ok=True)
    print("Downloading the pinned MM-TelcoBench document corpus...", flush=True)
    with requests.get(CORPUS_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for block in response.iter_content(1024 * 1024):
                if block:
                    handle.write(block)
    actual_corpus = sha256_file(temporary)
    if actual_corpus != CORPUS_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded corpus hash mismatch: expected {CORPUS_SHA256}, got {actual_corpus}"
        )
    temporary.replace(CORPUS_PATH)
    print(f"Corpus verified: {CORPUS_PATH}", flush=True)


def load_retriever_class():
    spec = importlib.util.spec_from_file_location("local_benchmark_adapter", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load adapter: {ADAPTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Retriever


def ollama_identity(model: str) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            ["ollama", "show", model, "--modelfile"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Ollama is not available. Install Ollama and pull qwen3:1.7b.") from exc
    if completed.returncode:
        raise RuntimeError(f"Ollama cannot inspect {model}. Run: ollama pull {model}")
    match = re.search(r"sha256-([0-9a-f]{64})", completed.stdout)
    if not match:
        raise RuntimeError(f"Ollama did not report a blob digest for {model}.")

    response = requests.get("http://127.0.0.1:11434/api/tags", timeout=30)
    response.raise_for_status()
    manifest = ""
    for item in response.json().get("models", []):
        if item.get("name") == model or item.get("model") == model:
            manifest = str(item.get("digest", ""))
            break
    if not re.fullmatch(r"[0-9a-f]{64}", manifest):
        raise RuntimeError(f"Ollama did not report a manifest digest for {model}.")
    return match.group(1), manifest


def verify_model(model: str, allow_model_mismatch: bool) -> tuple[str, str]:
    blob, manifest = ollama_identity(model)
    if model == DEFAULT_MODEL and not allow_model_mismatch:
        if blob != MODEL_BLOB_SHA256:
            raise RuntimeError(
                f"Model blob mismatch: expected {MODEL_BLOB_SHA256}, got {blob}. "
                "Use --allow-model-mismatch only when exact reproduction is not required."
            )
        if manifest != MODEL_MANIFEST_SHA256:
            raise RuntimeError(
                f"Model manifest mismatch: expected {MODEL_MANIFEST_SHA256}, got {manifest}. "
                "Use --allow-model-mismatch only when exact reproduction is not required."
            )
    return blob, manifest


def stop_model(model: str) -> None:
    try:
        subprocess.run(
            ["ollama", "stop", model],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def make_run_id(settings: dict) -> str:
    payload = json.dumps(settings, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_settings(args: argparse.Namespace, retrieval_only: bool) -> dict:
    blob = "not-used"
    manifest = "not-used"
    if not retrieval_only:
        blob, manifest = verify_model(args.model, args.allow_model_mismatch)
    return {
        "system": SYSTEM_ID,
        "display_name": DISPLAY_NAME,
        "qa_sha256": sha256_file(QA_PATH),
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "runner_sha256": sha256_file(ROOT / "run.py"),
        "adapter_sha256": sha256_file(ADAPTER_PATH),
        "runtime_sha256": sha256_file(ROOT / "common_runtime.py"),
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "embedding_device": "cpu",
        "embedding_batch_size": args.embedding_batch_size,
        "generator": args.model,
        "generator_blob_sha256": blob,
        "generator_manifest_sha256": manifest,
        "ollama_num_gpu_layers": args.num_gpu_layers,
        "ollama_num_thread": 2,
        "ollama_num_batch": 16,
        "temperature": 0,
        "seed": 42,
        "top_k": args.top_k,
        "options_used_for_retrieval": False,
        "retrieval_only": retrieval_only,
        "two_pass_retrieval": TWO_PASS_RETRIEVAL,
    }


def build_retriever(args: argparse.Namespace):
    rows = load_corpus(CORPUS_PATH)
    embedder = SharedEmbedder(
        args.embedding_model,
        CACHE_ROOT,
        CORPUS_PATH,
        batch_size=args.embedding_batch_size,
        revision=args.embedding_revision,
    )
    embeddings = embedder.corpus_embeddings(rows)
    retriever = load_retriever_class()(
        rows,
        embeddings,
        embedder.query,
        CACHE_ROOT / SYSTEM_ID,
    )
    return rows, retriever


def prepare(args: argparse.Namespace) -> None:
    ensure_inputs()
    rows, _ = build_retriever(args)
    questions = load_questions(QA_PATH)
    print(f"Prepared {DISPLAY_NAME}: {len(rows)} documents, {len(questions)} questions")


def load_completed(path: Path, run_id: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    completed = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") == run_id:
                completed[row["question_id"]] = row
    return completed


def run(args: argparse.Namespace, tag: str, limit: int | None) -> Path:
    ensure_inputs()
    questions = load_questions(QA_PATH)
    if limit is not None:
        questions = questions[:limit]

    retrieval_only = bool(args.retrieval_only)
    settings = build_settings(args, retrieval_only)
    _, retriever = build_retriever(args)
    run_id = make_run_id(settings | {"tag": tag, "limit": limit})
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_ROOT / f"{tag}.jsonl"
    completed = load_completed(raw_path, run_id)

    client = None
    if not retrieval_only:
        client = OllamaClient(
            model=args.model,
            cooldown_seconds=args.cooldown,
            num_gpu_layers=args.num_gpu_layers,
        )
        atexit.register(stop_model, args.model)

    try:
        for index, question in enumerate(questions, start=1):
            if question["id"] in completed:
                continue

            candidate_answer = ""
            candidate_seconds = 0.0
            retrieval_seconds = 0.0
            if TWO_PASS_RETRIEVAL:
                started = time.perf_counter()
                first = retriever.retrieve(question["question"], top_k=args.top_k, candidate_answer="")
                retrieval_seconds += time.perf_counter() - started
                if client is not None:
                    candidate_answer, candidate_seconds = client.candidate_answer(
                        question["question"], first["contexts"]
                    )
                    started = time.perf_counter()
                    retrieval = retriever.retrieve(
                        question["question"],
                        top_k=args.top_k,
                        candidate_answer=candidate_answer,
                    )
                    retrieval_seconds += time.perf_counter() - started
                else:
                    retrieval = first
            else:
                started = time.perf_counter()
                retrieval = retriever.retrieve(question["question"], top_k=args.top_k)
                retrieval_seconds = time.perf_counter() - started

            raw_answer = ""
            generation_seconds = 0.0
            predicted = None
            if client is not None:
                raw_answer, generation_seconds = client.answer_mcq(
                    question, retrieval["contexts"]
                )
                predicted = parse_choice(raw_answer)

            relevant = str(question["relevant_doc_id"])
            doc_ids = [str(value) for value in retrieval["doc_ids"]]
            record = {
                "run_id": run_id,
                "tag": tag,
                "system": SYSTEM_ID,
                "question_id": question["id"],
                "question": question["question"],
                "gold": question["answer"],
                "predicted": predicted,
                "correct": predicted == question["answer"] if predicted else False,
                "raw_answer": raw_answer,
                "candidate_answer": candidate_answer,
                "relevant_doc_id": relevant,
                "retrieved_doc_ids": doc_ids,
                "hit_at_1": relevant in doc_ids[:1],
                "hit_at_3": relevant in doc_ids[:3],
                "hit_at_5": relevant in doc_ids[:5],
                "retrieval_seconds": max(0.0, retrieval_seconds),
                "candidate_seconds": candidate_seconds,
                "generation_seconds": generation_seconds,
                "trace": retrieval.get("trace", {}),
                "settings": settings,
            }
            with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"[{index}/{len(questions)}] {question['id']} "
                f"answer={predicted or '-'} gold={question['answer']} hit5={record['hit_at_5']}",
                flush=True,
            )
    finally:
        if client is not None:
            stop_model(args.model)
            atexit.unregister(stop_model)

    rows = list(load_completed(raw_path, run_id).values())
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    low, high = wilson_interval(correct, total)
    summary = {
        "run_id": run_id,
        "tag": tag,
        "settings": settings,
        "metrics": {
            "n": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "accuracy_wilson_95": [low, high],
            "parse_failures": sum(row["predicted"] is None for row in rows),
            "retrieval_hit_at_1": sum(row["hit_at_1"] for row in rows) / total if total else 0.0,
            "retrieval_hit_at_3": sum(row["hit_at_3"] for row in rows) / total if total else 0.0,
            "retrieval_hit_at_5": sum(row["hit_at_5"] for row in rows) / total if total else 0.0,
            "mean_retrieval_seconds": statistics.fmean(row["retrieval_seconds"] for row in rows) if rows else 0.0,
            "mean_generation_seconds": statistics.fmean(row["generation_seconds"] for row in rows) if rows else 0.0,
            "mean_candidate_seconds": statistics.fmean(row["candidate_seconds"] for row in rows) if rows else 0.0,
            "generator_calls_per_question": 0 if retrieval_only else (2 if TWO_PASS_RETRIEVAL else 1),
        },
    }
    summary_path = RESULTS_ROOT / f"benchmark_{tag}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Reproduce the shared 100-question benchmark for {DISPLAY_NAME}."
    )
    parser.add_argument("command", choices=("prepare", "smoke", "full"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--embedding-revision",
        default="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-layers", type=int, default=8)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--allow-model-mismatch", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.embedding_batch_size <= 4:
        parser.error("embedding batch size must be between 1 and 4")
    if not 0 <= args.num_gpu_layers <= 8:
        parser.error("GPU layers must be between 0 and 8")
    if args.top_k < 5:
        parser.error("top-k must be at least 5 because Hit@5 is reported")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "smoke":
        run(args, "smoke", args.limit or 3)
    else:
        run(args, "full", args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
