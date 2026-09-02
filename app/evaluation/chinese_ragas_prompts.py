"""面向中文客服回答的 RAGAS claim 与 NLI prompts。"""
import json

from ragas.metrics._factual_correctness import ClaimDecompositionInput, ClaimDecompositionOutput
from ragas.metrics._faithfulness import (
    NLIStatementInput,
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorInput,
    StatementGeneratorOutput,
)
from ragas.prompt import PydanticPrompt


class ChineseJSONPrompt(PydanticPrompt):
    """将 RAGAS 默认英文 JSON 说明替换为中文。"""

    def _generate_output_signature(self, indent: int = 4) -> str:
        schema = json.dumps(self.output_model.model_json_schema(), ensure_ascii=False, indent=indent)
        return f"请严格按以下 JSON Schema 输出合法 JSON，不要附加解释或 Markdown：\n{schema}"


class ChineseStatementGeneratorPrompt(ChineseJSONPrompt):
    instruction = """请将中文客服回答拆分为可独立核验的事实陈述。保留金额、期限、条件、否定词和条款编号；不要生成寒暄、建议或问题。仅输出 JSON。"""
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    examples = [
        (
            StatementGeneratorInput(
                question="黑卡会员试穿过内衣还能退吗？",
                answer="不能退。依据 VIP_006 与 CAT_001，黑卡会员退贴身衣物仍须未使用、未洗涤；试穿属于已使用。",
            ),
            StatementGeneratorOutput(statements=[
                "黑卡会员退贴身衣物仍须未使用、未洗涤。",
                "内衣试穿属于已使用。",
                "试穿过的内衣不能退货。",
            ]),
        ),
        (
            StatementGeneratorInput(
                question="偏远地区包邮门槛是多少？",
                answer="新疆、西藏、青海订单满 199 元包邮；不满 199 元收取 12 元运费。",
            ),
            StatementGeneratorOutput(statements=[
                "新疆、西藏、青海订单满 199 元包邮。",
                "新疆、西藏、青海订单不满 199 元收取 12 元运费。",
            ]),
        ),
    ]


class ChineseNLIStatementPrompt(ChineseJSONPrompt):
    instruction = """根据给定上下文逐条核验陈述。只有上下文明确支持或可直接推得出时 verdict=1；缺少依据、数值或条件不一致、与上下文冲突时 verdict=0。中文措辞、空格和标点差异不构成冲突。仅输出 JSON。"""
    input_model = NLIStatementInput
    output_model = NLIStatementOutput
    examples = [
        (
            NLIStatementInput(
                context="CAT_001：贴身衣物一旦拆封、试穿或水洗，概不退换。未拆封且不影响二次销售的，7 天内可退。",
                statements=[
                    "试穿过的内衣不能退货。",
                    "未拆封的内衣在 7 天内可以退货。",
                    "试穿过的内衣可以申请极速退款。",
                ],
            ),
            NLIStatementOutput(statements=[
                StatementFaithfulnessAnswer(statement="试穿过的内衣不能退货。", reason="上下文明确规定贴身衣物试穿后概不退换。", verdict=1),
                StatementFaithfulnessAnswer(statement="未拆封的内衣在 7 天内可以退货。", reason="上下文明确给出未拆封且不影响二次销售时 7 天内可退。", verdict=1),
                StatementFaithfulnessAnswer(statement="试穿过的内衣可以申请极速退款。", reason="上下文没有支持试穿后可退或可极速退款，且与概不退换冲突。", verdict=0),
            ]),
        ),
        (
            NLIStatementInput(
                context="SHIP_003：包裹显示已签收但用户未收到时，应在 48 小时内联系客服核实，超过期限将难以追溯。",
                statements=[
                    "包裹显示签收但未收到时，应在 48 小时内联系客服。",
                    "包裹显示签收但未收到时，平台保证 24 小时内赔付。",
                ],
            ),
            NLIStatementOutput(statements=[
                StatementFaithfulnessAnswer(statement="包裹显示签收但未收到时，应在 48 小时内联系客服。", reason="上下文直接支持该时限和行动。", verdict=1),
                StatementFaithfulnessAnswer(statement="包裹显示签收但未收到时，平台保证 24 小时内赔付。", reason="上下文未提到赔付或 24 小时时限。", verdict=0),
            ]),
        ),
    ]


class ChineseClaimDecompositionPrompt(ChineseJSONPrompt):
    instruction = """将输入中文文本拆成原子事实陈述，分别保留主体、条件、数值、期限和否定信息。不得补充文本外事实；不拆分礼貌用语。仅输出 JSON。"""
    input_model = ClaimDecompositionInput
    output_model = ClaimDecompositionOutput
    examples = [
        (
            ClaimDecompositionInput(response="金卡会员享 30 天无理由退货，极速退款单笔不超过 2000 元，每月最多 3 次。"),
            ClaimDecompositionOutput(claims=[
                "金卡会员享有 30 天无理由退货。",
                "金卡会员极速退款单笔不超过 2000 元。",
                "金卡会员极速退款每月最多 3 次。",
            ]),
        ),
        (
            ClaimDecompositionInput(response="新鞋有轻微胶水味不构成质量问题，但符合商品完好等条件时仍可按 7 天无理由退货申请。"),
            ClaimDecompositionOutput(claims=[
                "新鞋有轻微胶水味不构成质量问题。",
                "商品符合完好等条件时可以按 7 天无理由退货申请。",
            ]),
        ),
    ]
