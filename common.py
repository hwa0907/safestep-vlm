"""공통 모듈: 프롬프트, 출력 스키마, JSON 파싱, Qwen2.5-VL 로딩/추론 헬퍼.

모든 단계(teacher 라벨링, 학습, 평가, 데모)가 같은 프롬프트와 스키마를 쓰도록
여기 한 곳에 모아 둔다. 프롬프트를 바꾸면 teacher 라벨과 student 학습이 어긋나므로
바꿀 때는 라벨링부터 다시 해야 한다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ----------------------------------------------------------------------------
# 출력 스키마
# ----------------------------------------------------------------------------
HAZARD_VOCAB = [
    "사람", "자전거", "전동킥보드", "오토바이", "자동차",
    "볼라드", "기둥", "가로수", "바리케이드", "계단", "턱/단차", "공사구역",
    "웅덩이", "횡단보도", "신호등", "점자블록", "입간판",
    "쓰레기통", "동물", "기타",
]
RISK_LEVELS = ["높음", "보통", "낮음"]
POSITIONS = ["왼쪽", "가운데", "오른쪽"]
DISTANCES = ["가까움", "보통", "멀리"]

SYSTEM_PROMPT = "당신은 시각장애인의 안전한 보행을 돕는 음성 안내 도우미입니다."

USER_PROMPT = (
    "이 사진은 보행자 시점에서 촬영한 장면입니다. 보행에 영향을 주는 위험 요소를 찾아 "
    "아래 JSON 형식으로만 답하세요. 다른 설명은 쓰지 마세요.\n"
    '{"risk": "높음|보통|낮음", '
    '"hazards": [{"type": "<종류>", "position": "왼쪽|가운데|오른쪽", "distance": "가까움|보통|멀리"}], '
    '"guidance": "<한두 문장의 한국어 음성 안내>"}\n'
    "- type은 다음 중 하나: " + ", ".join(HAZARD_VOCAB) + " (승용차·버스·트럭은 모두 자동차)\n"
    "- risk 기준: 높음 = 진행 방향(가운데)에 가까운 장애물이 있거나, 가까이에 자동차·오토바이·자전거·킥보드가 있음 / "
    "보통 = 가까운 장애물이 옆(왼쪽·오른쪽)에 있거나, 가운데에 보통 거리의 장애물이 있음 / "
    "낮음 = 멀리 있는 것만 있거나 아무것도 없음\n"
    "- hazards에는 보행에 영향을 주는 것만 가까운 순서로 최대 3개 쓰세요. 멀리 있는 것은 가운데에 있을 때만 쓰고, "
    "없으면 빈 리스트로 두세요.\n"
    "- guidance는 한두 문장으로, 가장 중요한 위험을 '방향 + 거리 + 물체' 순서로 먼저 말하고 "
    "구체적인 행동(멈추세요 / 왼쪽으로 비켜 가세요 / 그대로 직진하세요 등)을 말하세요. "
    "위험이 낮으면 '앞이 비어 있습니다. 그대로 직진하세요.'처럼 쓰세요."
)


def teacher_prompt(hint_text: str | None) -> str:
    """Teacher에게만 자동 검출 힌트를 덧붙인다 (student는 이미지만 보고 학습)."""
    if not hint_text:
        return USER_PROMPT
    return (
        USER_PROMPT
        + "\n\n참고용 자동 검출 결과(틀릴 수 있으니 사진을 직접 확인하세요): "
        + hint_text
    )


# ----------------------------------------------------------------------------
# 출력 파싱 / 정규화
# ----------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


def parse_output(text: str) -> dict | None:
    """모델 출력 문자열 → 정규화된 dict. 형식이 깨졌으면 None."""
    if not text:
        return None
    s = _FENCE.sub("", text).strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return normalize_label(obj)


def normalize_label(obj: dict) -> dict | None:
    risk = obj.get("risk")
    guidance = obj.get("guidance")
    hazards = obj.get("hazards", [])
    if risk not in RISK_LEVELS or not isinstance(guidance, str) or not guidance.strip():
        return None
    if not isinstance(hazards, list):
        return None
    clean = []
    for h in hazards:
        if not isinstance(h, dict):
            continue
        t = h.get("type")
        clean.append({
            "type": t if t in HAZARD_VOCAB else "기타",
            "position": h.get("position") if h.get("position") in POSITIONS else "가운데",
            "distance": h.get("distance") if h.get("distance") in DISTANCES else "보통",
        })
    return {"risk": risk, "hazards": clean, "guidance": guidance.strip()}


def dumps(label: dict) -> str:
    return json.dumps(label, ensure_ascii=False)


# ----------------------------------------------------------------------------
# jsonl 유틸
# ----------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# Qwen2.5-VL 로딩 / 추론 (torch, transformers는 여기서만 import → CPU 전용 단계는 가볍게)
# ----------------------------------------------------------------------------
def build_messages(prompt: str, answer: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def pick_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # T4(Colab 무료)는 bf16 미지원 → fp16
    return torch.float16


def load_processor(model_id: str, max_pixels: int, min_pixels: int = 128 * 28 * 28):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)


def load_model(model_id: str, load_4bit: bool = True, adapter: str | None = None):
    import torch
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    dtype = pick_dtype()
    kwargs = dict(torch_dtype=dtype, device_map="auto")
    if load_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def generate(model, processor, image, prompt: str = USER_PROMPT, max_new_tokens: int = 256) -> str:
    """이미지 1장 → 모델 출력 문자열 (greedy)."""
    import torch
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    text = processor.apply_chat_template(build_messages(prompt), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
