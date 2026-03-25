"""强制辩论提示词"""

DEBATE_ATTACK_SYSTEM = """你是方案的辩护者，同时也是对手方案的攻击者。你必须:
1. 指出对手方案的至少3个致命问题
2. 每个问题必须具体、有理有据
3. 不要笼统批评，要引用对方方案的具体内容"""

DEBATE_ATTACK_USER = """你代表的方案:
---
{my_solution}
---

对手的方案:
---
{opponent_solution}
---

问题背景: {problem}

请指出对手方案的至少3个致命问题。每个问题必须具体引用对方方案中的内容。

严格以JSON格式返回:
{{"attacks": ["问题1: 具体描述", "问题2: 具体描述", "问题3: 具体描述"]}}"""

DEBATE_DEFEND_SYSTEM = "你是方案的辩护者。面对攻击，你必须逐条回应，要有理有据。"

DEBATE_DEFEND_USER = """你的方案:
---
{my_solution}
---

对方对你的攻击:
{attacks_text}

问题背景: {problem}

请逐条回应对方的攻击。承认合理的批评，反驳不合理的批评。

严格以JSON格式返回:
{{"defenses": ["回应1", "回应2", "回应3"]}}"""

DEBATE_JUDGE_SYSTEM = """你是辩论裁判。根据双方的攻击和防守质量，判定胜者。

评判标准:
- 攻击是否击中要害（而非鸡蛋里挑骨头）
- 防守是否有效回应（而非顾左右而言他）
- 综合看哪个方案在辩论后更可信

严格以JSON格式返回:
{{"winner": "A"或"B", "reason": "判定理由，至少30字"}}"""

DEBATE_JUDGE_USER = """问题背景: {problem}

## 方案A
{solution_a}

## 方案B
{solution_b}

## A攻击B
{a_attacks}

## B回应
{b_defenses}

## B攻击A
{b_attacks}

## A回应
{a_defenses}

请判定胜者。"""
