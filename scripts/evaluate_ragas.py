"""对 baseline/oracle 运行产物执行 RAGAS 0.4+ 自动评估。"""
import argparse
import json
from math import isnan
from pathlib import Path
import sys
from types import ModuleType

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _install_vertexai_import_compatibility() -> None:
    """兼容 RAGAS 0.4.3 对已移除 LangChain VertexAI 路径的无条件导入。

    本项目使用 Qwen/OpenAI 兼容接口，不会实例化这些占位类型。
    """
    import langchain_community.llms as community_llms

    class UnavailableVertexAI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("当前评估未配置 VertexAI")

    module_name = "langchain_community.chat_models.vertexai"
    if module_name not in sys.modules:
        module = ModuleType(module_name)
        module.ChatVertexAI = UnavailableVertexAI
        sys.modules[module_name] = module
    if not hasattr(community_llms, "VertexAI"):
        community_llms.VertexAI = UnavailableVertexAI


def _require_ragas():
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness
        from ragas.run_config import RunConfig
    except ModuleNotFoundError as exc:
        if exc.name not in {"langchain_community.chat_models.vertexai"}:
            raise
        _install_vertexai_import_compatibility()
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError(
            "未安装 RAGAS。请先运行 `uv sync --group dev`，再执行本脚本。"
        ) from exc
    return {
        "EvaluationDataset": EvaluationDataset,
        "SingleTurnSample": SingleTurnSample,
        "evaluate": evaluate,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "Faithfulness": Faithfulness,
        "RunConfig": RunConfig,
    }


def main(input_path: Path, output_path: Path, limit: int | None, max_workers: int) -> None:
    ragas = _require_ragas()
    from app.evaluation.chinese_ragas_prompts import (
        ChineseNLIStatementPrompt, ChineseStatementGeneratorPrompt,
    )
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr
    from app.core.config import settings

    if not settings.JUDGE_LLM_MODEL or not settings.JUDGE_OPENAI_API_KEY:
        raise RuntimeError("请在 .env 配置 JUDGE_LLM_MODEL 与 JUDGE_OPENAI_API_KEY，再运行 RAGAS 评估。")
    if settings.JUDGE_LLM_MODEL == settings.LLM_MODEL:
        raise RuntimeError("JUDGE_LLM_MODEL 必须与 LLM_MODEL 不同，避免业务模型自评。")

    run = json.loads(input_path.read_text(encoding="utf-8"))
    run_items = run["items"][:limit] if limit else run["items"]
    samples = [ragas["SingleTurnSample"](
        user_input=item["question"],
        retrieved_contexts=[context["content"] for context in item["retrieved_contexts"]] + item.get("policy_rules", []),
        response=item["answer"],
        reference=item["ground_truth"],
    ) for item in run_items]
    dataset = ragas["EvaluationDataset"](samples=samples)

    judge_llm = ChatOpenAI(
        base_url=settings.JUDGE_OPENAI_BASE_URL or settings.OPENAI_BASE_URL,
        api_key=SecretStr(settings.JUDGE_OPENAI_API_KEY),
        model=settings.JUDGE_LLM_MODEL,
        temperature=0,
        # DeepSeek V4（v4-flash/v4-pro）默认开启思维链（thinking），会对结构化 JSON
        # 输出 prompt 生成数百 reasoning tokens，单次调用从 <1s 暴涨到 10~30s 且波动极大，
        # 叠加 RAGAS 每题 4 次调用后频繁触发 RunConfig(timeout=120) 超时。
        # judge 是确定性判断，无需推理，禁用 thinking 后单次回到 ~0.7s。
        # 若更换为非 DeepSeek judge，需移除该参数。
        extra_body={"thinking": {"type": "disabled"}},
    )
    # 只保留 faithfulness：衡量「答案是否忠于检索上下文、不编造」。
    # 曾同时评测 factual_correctness(mode=precision)，但该模式只罚「答案里有而 ground_truth
    # 没有的 claim」，客服答案天然比一句话 ground_truth 长（条款引用+客套话+追问），导致
    # 全对答案被系统性判低分（4 个 0 分题答案与 ground_truth 完全一致），已移除。
    # 生成侧的事实正确性改由确定性的 answer_clause_metrics（run_rag_baseline.py）衡量。
    metrics = [
        ragas["Faithfulness"](
            statement_generator_prompt=ChineseStatementGeneratorPrompt(),
            nli_statements_prompt=ChineseNLIStatementPrompt(),
            max_retries=2,
        ),
    ]
    run_config = ragas["RunConfig"](timeout=120, max_retries=3, max_workers=max_workers)
    result = ragas["evaluate"](
        dataset=dataset,
        metrics=metrics,
        llm=ragas["LangchainLLMWrapper"](judge_llm),
        run_config=run_config,
    )
    frame = result.to_pandas()
    records = frame.to_dict(orient="records")
    def is_missing(value) -> bool:
        return value is None or (isinstance(value, float) and isnan(value))

    null_counts = {column: sum(is_missing(row.get(column)) for row in records) for column in frame.columns}
    report = {
        "input": str(input_path),
        "evaluated_count": len(records),
        "judge": {"model": settings.JUDGE_LLM_MODEL, "base_url": settings.JUDGE_OPENAI_BASE_URL or settings.OPENAI_BASE_URL},
        "metrics": [metric.name for metric in metrics],
        "run_config": {"timeout": 120, "max_retries": 3, "max_workers": max_workers},
        "null_counts": null_counts,
        "items": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"RAGAS 结果已写入 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="scripts/run_rag_baseline.py 生成的运行产物")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="先运行标定子集，例如 15")
    parser.add_argument("--max-workers", type=int, default=2, help="judge 并发数，默认 2")
    args = parser.parse_args()
    default_output = args.input.with_name(f"{args.input.stem}.ragas.json")
    main(args.input, args.output or default_output, args.limit, args.max_workers)
