"""运行可复现的真实检索或 oracle-context RAG 基线。"""
import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from app.evaluation.metrics import (
    aggregate_by, aggregate_metrics, answer_clause_metrics, clause_metrics, cosine_similarity,
)
from app.core.config import settings
from app.graph.nodes import GENERATE_SYSTEM_PROMPT, llm
from app.services.policy_retrieval import embedding_model, load_oracle_contexts, retrieve_policy


def serialize_context(context) -> dict:
    return {
        "content": context.content,
        "source": context.source,
        "clause_ids": context.clause_ids,
        "rank": context.rank,
        "distance": context.distance,
    }


async def generate_answer(question: str, contexts: list[dict]) -> str:
    evidence = "\n\n".join(
        f"【{', '.join(context['clause_ids']) or context['source']}】\n{context['content']}"
        for context in contexts
    ) or "暂无相关参考信息。"
    response = await llm.ainvoke([
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=f"[参考信息]：\n{evidence}\n\n[用户问题]：\n{question}"),
    ])
    return str(response.content)


async def answer_semantic_similarity(answer: str, ground_truth: str) -> float | None:
    try:
        answer_vector, reference_vector = await embedding_model.aembed_documents([answer, ground_truth])
        return cosine_similarity(answer_vector, reference_vector)
    except Exception:
        # 语义相似度是补充指标，单独失败不应丢弃该题主评测产物。
        return None


def checkpoint_path(output: Path) -> Path:
    return output.with_suffix(".jsonl")


def load_completed(checkpoint: Path) -> dict[str, dict]:
    if not checkpoint.exists():
        return {}
    completed: dict[str, dict] = {}
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") == "ok":
            completed[record["id"]] = record["item"]
    return completed


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


async def main(use_oracle: bool, output: Path, limit: int | None = None, concurrency: int = 4) -> None:
    dataset = json.loads(Path("eval/testset.json").read_text(encoding="utf-8"))
    test_items = dataset["items"][:limit] if limit else dataset["items"]
    semaphore = asyncio.Semaphore(concurrency)
    checkpoint = checkpoint_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(checkpoint)
    write_lock = asyncio.Lock()

    async def checkpoint_result(record: dict) -> None:
        async with write_lock:
            with checkpoint.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def evaluate_item(index: int, item: dict) -> dict | None:
        if item["id"] in completed:
            print(f"[{index}/{len(test_items)}] {item['id']} 已从 checkpoint 恢复", flush=True)
            return completed[item["id"]]
        started_at = datetime.now(timezone.utc)
        try:
            async with semaphore:
                print(f"[{index}/{len(test_items)}] {item['id']} 检索与生成中...", flush=True)
                retrieved = await (
                    load_oracle_contexts(item["expected_sources"])
                    if use_oracle else retrieve_policy(item["question"])
                )
                contexts = [serialize_context(context) for context in retrieved]
                answer = await generate_answer(item["question"], contexts)
                semantic_similarity = await answer_semantic_similarity(answer, item["ground_truth"])
            result = {
                **item,
                "retrieved_contexts": contexts,
                "answer": answer,
                "retrieval_metrics": clause_metrics(item["expected_sources"], contexts, k=5),
                "answer_clause_metrics": answer_clause_metrics(answer, item["expected_sources"]),
                "answer_semantic_similarity": semantic_similarity,
                "duration_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
            }
            await checkpoint_result({"status": "ok", "id": item["id"], "item": result})
            return result
        except Exception as exc:
            error = {
                "status": "error", "id": item["id"], "question": item["question"],
                "error_type": type(exc).__name__, "error": str(exc),
                "duration_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
            }
            await checkpoint_result(error)
            print(f"[{index}/{len(test_items)}] {item['id']} 失败: {error['error_type']}", flush=True)
            return None

    results = await asyncio.gather(*(evaluate_item(index, item) for index, item in enumerate(test_items, 1)))
    items = [item for item in results if item is not None]

    report = {
        "run_type": "oracle" if use_oracle else "baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "retrieval_config": {"top_k": 5, "similarity_threshold": 0.5},
        "models": {"llm": settings.LLM_MODEL, "embedding": settings.EMBEDDING_MODEL},
        "packages": {"langchain-openai": package_version("langchain-openai"), "ragas": package_version("ragas")},
        "checkpoint": str(checkpoint),
        "completed_count": len(items),
        "failed_count": len(test_items) - len(items),
        "summary": aggregate_metrics(items, k=5),
        "summary_by_category": aggregate_by(items, "category", k=5),
        "summary_by_difficulty": aggregate_by(items, "difficulty", k=5),
        "items": items,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评估运行结果已写入 {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true", help="使用 expected_sources 原文，隔离生成器质量")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="仅运行前 N 题，用于调试")
    parser.add_argument("--concurrency", type=int, default=4, help="并发请求数，默认 4")
    args = parser.parse_args()
    default_name = "oracle.json" if args.oracle else "baseline.json"
    asyncio.run(main(args.oracle, args.output or Path("eval/runs") / default_name, args.limit, args.concurrency))
