"""方案对话模式 — 进化完成后的交互式追问

包含对话历史的滑动窗口管理，防止超出模型 context window。
"""

from evohive.llm import call_llm
from evohive.prompts.dialogue_prompts import DIALOGUE_SYSTEM, DIALOGUE_USER

# 对话历史的最大轮次 (每轮 = 1 user + 1 assistant)
MAX_DIALOGUE_TURNS = 10


def _truncate_history(chat_history: list[dict], max_turns: int = MAX_DIALOGUE_TURNS) -> list[dict]:
    """截断对话历史，保留最近 max_turns 轮对话。

    滑动窗口策略: 保留最早的 1 轮 (上下文锚点) + 最近的 (max_turns-1) 轮。
    每条消息内容截断到 2000 字符以控制 token 用量。
    """
    if not chat_history:
        return []

    # 按对话轮次分组 (每 2 条为一轮: user + assistant)
    pairs = []
    for i in range(0, len(chat_history) - 1, 2):
        pair = chat_history[i:i+2]
        if len(pair) == 2:
            pairs.append(pair)

    if len(pairs) <= max_turns:
        # 不需要截断，但仍然截短内容
        return [
            {**msg, "content": msg["content"][:2000]}
            for msg in chat_history
        ]

    # 保留第 1 轮 + 最近 (max_turns-1) 轮
    kept_pairs = pairs[:1] + pairs[-(max_turns - 1):]
    result = []
    for pair in kept_pairs:
        for msg in pair:
            result.append({**msg, "content": msg["content"][:2000]})

    return result


async def dialogue_turn(
    user_message: str,
    solution_content: str,
    problem: str,
    chat_history: list[dict],
    model: str,
) -> str:
    """处理一轮对话

    Args:
        user_message: 用户的追问
        solution_content: 进化最优方案的完整内容
        problem: 原始问题
        chat_history: 之前的对话历史 [{"role": "user"/"assistant", "content": str}]
        model: 使用的LLM模型

    Returns:
        助手的回复文本
    """
    system = DIALOGUE_SYSTEM.format(
        problem=problem,
        solution_content=solution_content[:6000],  # 截断方案内容防止溢出
    )

    # Build message list with truncated history
    truncated = _truncate_history(chat_history)
    messages = [{"role": "system", "content": system}]
    for msg in truncated:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    # Use litellm directly for multi-turn conversation (with provider resolution)
    import litellm
    from evohive.llm.provider import _resolve_model

    try:
        resolved = _resolve_model(model)
        response = await litellm.acompletion(
            **resolved,
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"对话失败: {e}"
