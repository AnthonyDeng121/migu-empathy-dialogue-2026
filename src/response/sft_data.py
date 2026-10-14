"""Convert official raw conversations into supervised fine-tuning samples.

Training samples come either from train rows (turns + prediction_points with
gold targets) or from the frozen B1 response JSONL produced by
scripts/prepare_training_data.py --mode full (build_sft_samples_from_b1).
Evaluation samples come from the eval input rows produced by
scripts/make_eval_input.py (id / conversation_id / target_user_turn_id /
history) or from B1 response rows (context_turns), so training and inference
share the same prompt structure.

Leakage contract: the context never includes the assistant response being
predicted — history is cut at target_user_turn_id, and the supervised label is
target.response_text. memory_refs stays out of the prompt: the official
materials provide no memory bank, so the final candidate always outputs [].
"""
from __future__ import annotations

from typing import Any

BASE_SYSTEM_PROMPT = (
    "你是数字人综合情感陪伴对话助手。阅读对话历史，给出一段有共情的中文回复。"
    "只输出回复正文本身，不要输出任何解释、标签或引用编号。"
)

CONDITION_HEADER = "[当前用户状态参考]"


def profile_text(user_profile: dict[str, list[str]]) -> str:
    parts = [
        _enum_list("性格", user_profile.get("personality_traits", [])),
        _enum_list("兴趣", user_profile.get("interests", [])),
        _enum_list("风格", user_profile.get("style", [])),
    ]
    return "；".join(parts)


def _enum_list(label: str, values: list[str]) -> str:
    return f"{label}：{'、'.join(values) if values else '无'}"


def system_prompt(condition: dict[str, Any] | None) -> str:
    if not condition:
        return BASE_SYSTEM_PROMPT
    emotion = condition.get("emotion_label")
    profile = condition.get("user_profile")
    lines = [f"情感状态：{emotion if emotion else '无'}"]
    if profile is not None:
        lines.append(f"用户画像：{profile_text(profile)}")
    return f"{BASE_SYSTEM_PROMPT}\n{CONDITION_HEADER}{'；'.join(lines)}"


def _turns_by_id(turns: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(turn["turn_id"]): turn for turn in turns}


def history_messages(
    turns: list[dict[str, Any]],
    target_user_turn_id: int,
    max_history_turns: int | None = None,
) -> list[dict[str, str]]:
    """Chat messages for turns with turn_id <= target_user_turn_id.

    The target turn must be a user turn and stays last; the assistant response
    turn and any later turns are excluded by the cutoff. With max_history_turns
    the most recent turns are kept, always including the target turn.
    """
    indexed = _turns_by_id(turns)
    target = indexed.get(target_user_turn_id)
    if target is None:
        raise ValueError(f"target_user_turn_id {target_user_turn_id} not found in turns")
    if target["role"] != "user":
        raise ValueError(f"target_user_turn_id {target_user_turn_id} is not a user turn")
    selected = [
        turn
        for turn in turns
        if int(turn["turn_id"]) <= target_user_turn_id
    ]
    if max_history_turns is not None and len(selected) > max_history_turns:
        selected = selected[-max_history_turns:]
    return [{"role": turn["role"], "content": turn["content"]} for turn in selected]


def _condition_from_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "emotion_label": target.get("emotion_label"),
        "user_profile": target.get("user_profile"),
    }


B1_RESPONSE_FIELDS = (
    "sample_id",
    "conversation_id",
    "target_user_turn_id",
    "context_turns",
    "response_text",
)


def _b1_condition_index(condition_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Gold conditions keyed by sample_id (B1 classification JSONL)."""
    conditions: dict[str, dict[str, Any]] = {}
    for row in condition_rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("condition row without sample_id")
        emotion = row.get("emotion_label")
        profile = row.get("user_profile")
        if not isinstance(emotion, str) or not isinstance(profile, dict):
            raise ValueError(
                f"condition row {sample_id} lacks emotion_label/user_profile"
            )
        if sample_id in conditions:
            raise ValueError(f"duplicate condition row: {sample_id}")
        conditions[sample_id] = {"emotion_label": emotion, "user_profile": profile}
    if not conditions:
        raise ValueError("condition rows are empty")
    return conditions


def build_sft_samples_from_b1(
    rows: list[dict[str, Any]],
    condition_rows: list[dict[str, Any]] | None = None,
    condition_mode: str = "gold",
    max_history_turns: int | None = None,
) -> list[dict[str, Any]]:
    """SFT samples from frozen B1 response JSONL rows, in file order.

    B1 rows carry sample_id / context_turns / response_text; the conversion
    already excluded the predicted assistant turn and everything after it.
    The leak contract is re-verified here rather than trusted: context turn
    ids must be strictly ascending, end at the target user turn, and never
    exceed it, so the target assistant response (target_user_turn_id + 1)
    cannot appear in the context.

    condition_mode "gold" joins the B1 classification rows by sample_id so
    training keeps the emotion/profile conditioning of the D1 template; a
    missing condition is an error, never a silent fallback to the base
    prompt, which would mix conditioned and unconditioned training.
    """
    if condition_mode not in ("gold", "none"):
        raise ValueError(f"unknown condition_mode: {condition_mode}")
    conditions: dict[str, dict[str, Any]] | None = None
    if condition_mode == "gold":
        if condition_rows is None:
            raise ValueError(
                "condition_mode=gold requires condition_rows (B1 classification JSONL)"
            )
        conditions = _b1_condition_index(condition_rows)
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        for field in B1_RESPONSE_FIELDS:
            if field not in row:
                raise ValueError(f"B1 response row missing field: {field}")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if sample_id in seen:
            raise ValueError(f"duplicate B1 sample_id: {sample_id}")
        seen.add(sample_id)
        conversation_id = row["conversation_id"]
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"{sample_id}: conversation_id must be a non-empty string")
        target_user_turn_id = row["target_user_turn_id"]
        if type(target_user_turn_id) is not int or target_user_turn_id < 1:
            raise ValueError(f"{sample_id}: target_user_turn_id must be a positive integer")
        context_turns = row["context_turns"]
        if not isinstance(context_turns, list) or not context_turns:
            raise ValueError(f"{sample_id}: context_turns must be a non-empty list")
        response_text = row["response_text"]
        if not isinstance(response_text, str) or not response_text.strip():
            raise ValueError(f"empty response_text for {sample_id}")
        turn_ids: list[int] = []
        for turn in context_turns:
            if not isinstance(turn, dict):
                raise ValueError(f"{sample_id}: context turn is not an object")
            turn_id = turn.get("turn_id")
            if type(turn_id) is not int or turn_id < 1:
                raise ValueError(f"{sample_id}: context turn_id must be a positive integer")
            if turn.get("role") not in ("user", "assistant"):
                raise ValueError(f"{sample_id}: context role must be user or assistant")
            if "content" not in turn:
                raise ValueError(f"{sample_id}: context turn missing content")
            turn_ids.append(turn_id)
        if turn_ids != sorted(turn_ids) or len(set(turn_ids)) != len(turn_ids):
            raise ValueError(f"{sample_id}: context turn_ids must be strictly ascending")
        if turn_ids[-1] != target_user_turn_id:
            raise ValueError(f"{sample_id}: last context turn must be the target user turn")
        if context_turns[-1]["role"] != "user":
            raise ValueError(f"{sample_id}: target turn must be a user turn")
        condition = None
        if conditions is not None:
            condition = conditions.get(sample_id)
            if condition is None:
                raise ValueError(
                    f"{sample_id}: no gold condition row; refusing to mix "
                    "conditioned and unconditioned prompts"
                )
        messages = [
            {"role": "system", "content": system_prompt(condition)},
            *history_messages(context_turns, target_user_turn_id, max_history_turns),
        ]
        samples.append(
            {
                "id": sample_id,
                "conversation_id": conversation_id,
                "target_user_turn_id": target_user_turn_id,
                "messages": messages,
                "response_text": response_text,
            }
        )
    return samples


def build_sft_samples(
    rows: list[dict[str, Any]],
    condition_mode: str = "gold",
    max_history_turns: int | None = None,
) -> list[dict[str, Any]]:
    """One supervised sample per prediction point, in file order.

    condition_mode: "gold" injects the point's own emotion/profile into the
    system prompt (training-time conditioning); "none" uses the base prompt.
    Predicted conditions at inference time are supplied by the caller.
    """
    if condition_mode not in ("gold", "none"):
        raise ValueError(f"unknown condition_mode: {condition_mode}")
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        conversation_id = row["conversation_id"]
        turns = row["turns"]
        turn_contents = _turns_by_id(turns)
        for point in row.get("prediction_points", []):
            target_user_turn_id = int(point["target_user_turn_id"])
            sample_id = f"{conversation_id}_t{target_user_turn_id}"
            if sample_id in seen:
                raise ValueError(f"duplicate prediction point: {sample_id}")
            seen.add(sample_id)
            response_text = point["target"]["response_text"]
            if not response_text or not response_text.strip():
                raise ValueError(f"empty response_text for {sample_id}")
            condition = (
                _condition_from_target(point["target"]) if condition_mode == "gold" else None
            )
            messages = [
                {"role": "system", "content": system_prompt(condition)},
                *history_messages(turns, target_user_turn_id, max_history_turns),
            ]
            samples.append(
                {
                    "id": sample_id,
                    "conversation_id": conversation_id,
                    "target_user_turn_id": target_user_turn_id,
                    "assistant_response_turn_id": int(point["assistant_response_turn_id"]),
                    "messages": messages,
                    "response_text": response_text,
                }
            )
            # Keep the response turn reference honest: the label must come from
            # the official target field, not from reconstructing the transcript.
            if int(point["assistant_response_turn_id"]) not in turn_contents:
                raise ValueError(f"assistant_response_turn_id missing for {sample_id}")
    return samples


def eval_row_id(row: dict[str, Any]) -> str:
    """The sample id of an eval input row (official test format or B1)."""
    for field in ("id", "sample_id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise ValueError("eval row needs id or sample_id")


def build_eval_sample(
    row: dict[str, Any],
    condition: dict[str, Any] | None = None,
    max_history_turns: int | None = None,
) -> dict[str, Any]:
    """Messages for an eval input row.

    Accepts both eval-input rows (id / target_user_turn_id / history, the
    official test format) and B1 response rows (sample_id / context_turns),
    so the same inference path serves the frozen val split and the official
    test file.
    """
    history = row.get("history")
    if history is None:
        history = row.get("context_turns")
    if history is None:
        raise ValueError("eval row needs history or context_turns")
    target_user_turn_id = int(row["target_user_turn_id"])
    return {
        "id": eval_row_id(row),
        "conversation_id": row.get("conversation_id"),
        "target_user_turn_id": target_user_turn_id,
        "messages": [
            {"role": "system", "content": system_prompt(condition)},
            *history_messages(history, target_user_turn_id, max_history_turns),
        ],
    }


def response_turn_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How often target.response_text equals the assistant turn transcript.

    Both fields exist in train data; the label is target.response_text, and a
    mismatch only means the transcript differs from the official target.
    """
    total = 0
    match = 0
    mismatch_examples: list[str] = []
    for row in rows:
        contents = _turns_by_id(row["turns"])
        for point in row.get("prediction_points", []):
            total += 1
            response_id = int(point["assistant_response_turn_id"])
            response_turn = contents.get(response_id)
            if response_turn is not None and response_turn["content"] == point["target"]["response_text"]:
                match += 1
            elif len(mismatch_examples) < 20:
                mismatch_examples.append(f"{row['conversation_id']}_t{point['target_user_turn_id']}")
    return {
        "total": total,
        "match": match,
        "mismatch": total - match,
        "mismatch_examples": mismatch_examples,
    }
