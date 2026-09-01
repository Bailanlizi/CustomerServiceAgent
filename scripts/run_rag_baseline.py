"""运行可复现的真实检索或 oracle-context RAG 基线。"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from app.evaluation.metrics import aggregate_metrics, clause_metrics
from app.graph.nodes import GENERATE_SYSTEM_PROMPT, llm
from app.services.policy_retrieval import load_oracle_contexts, retrieve_policy


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


async def main(use_oracle: bool, output: Path, limit: int | None = None, concurrency: int = 4) -> None:
    dataset = json.loads(Path("eval/testset.json").read_text(encoding="utf-8"))
    test_items = dataset["items"][:limit] if limit else dataset["items"]
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_item(index: int, item: dict) -> dict:
        async with semaphore:
            print(f"[{index}/{len(test_items)}] {item['id']} 检索与生成中...", flush=True)
            retrieved = await (
                load_oracle_contexts(item["expected_sources"])
                if use_oracle else retrieve_policy(item["question"])
            )
            contexts = [serialize_context(context) for context in retrieved]
            answer = await generate_answer(item["question"], contexts)
        return {
            **item,
            "retrieved_contexts": contexts,
            "answer": answer,
            "retrieval_metrics": clause_metrics(item["expected_sources"], contexts, k=5),
        }

    items = await asyncio.gather(*(evaluate_item(index, item) for index, item in enumerate(test_items, 1)))

    report = {
        "run_type": "oracle" if use_oracle else "baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "retrieval_config": {"top_k": 5, "similarity_threshold": 0.5},
        "summary": aggregate_metrics(items, k=5),
        "items": items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
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
