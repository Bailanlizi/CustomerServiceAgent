"""对 baseline/oracle 运行产物执行 RAGAS 0.4+ 自动评估。"""
import argparse
import json
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
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            ContextRelevance,
            Faithfulness,
            FactualCorrectness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"langchain_community.chat_models.vertexai"}:
            raise
        _install_vertexai_import_compatibility()
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            ContextRelevance,
            Faithfulness,
            FactualCorrectness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise RuntimeError(
            "未安装 RAGAS。请先运行 `uv sync --group dev`，再执行本脚本。"
        ) from exc
    return {
        "EvaluationDataset": EvaluationDataset,
        "SingleTurnSample": SingleTurnSample,
        "evaluate": evaluate,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "metrics": [
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            ContextRelevance(),
            Faithfulness(),
            ResponseRelevancy(),
            FactualCorrectness(),
        ],
    }


def main(input_path: Path, output_path: Path) -> None:
    ragas = _require_ragas()
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from pydantic import SecretStr
    from app.core.config import settings

    run = json.loads(input_path.read_text(encoding="utf-8"))
    samples = [ragas["SingleTurnSample"](
        user_input=item["question"],
        retrieved_contexts=[context["content"] for context in item["retrieved_contexts"]],
        response=item["answer"],
        reference=item["ground_truth"],
    ) for item in run["items"]]
    dataset = ragas["EvaluationDataset"](samples=samples)

    judge_llm = ChatOpenAI(
        base_url=settings.OPENAI_BASE_URL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        model=settings.LLM_MODEL,
        temperature=0,
    )
    judge_embeddings = OpenAIEmbeddings(
        base_url=settings.OPENAI_BASE_URL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        model=settings.EMBEDDING_MODEL,
        check_embedding_ctx_length=False,
    )
    result = ragas["evaluate"](
        dataset=dataset,
        metrics=ragas["metrics"],
        llm=ragas["LangchainLLMWrapper"](judge_llm),
        embeddings=ragas["LangchainEmbeddingsWrapper"](judge_embeddings),
    )
    frame = result.to_pandas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(output_path, orient="records", force_ascii=False, indent=2)
    print(f"RAGAS 结果已写入 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="scripts/run_rag_baseline.py 生成的运行产物")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    default_output = args.input.with_name(f"{args.input.stem}.ragas.json")
    main(args.input, args.output or default_output)
