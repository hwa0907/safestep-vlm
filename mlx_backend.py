"""Apple Silicon(MLX) 백엔드: 모델 로딩과 추론.

AI Hub 데이터는 해외 반출이 제한돼서(약관 3항) 학습·추론을 전부 로컬 맥에서 한다.
PyTorch + bitsandbytes(4bit)는 CUDA 전용이라 맥에서 못 쓰므로,
Apple의 MLX 프레임워크와 mlx-vlm 라이브러리로 같은 Qwen2.5-VL을 4bit로 돌린다.

학습·평가·데모가 같은 해상도를 보도록, 이미지는 항상 prep_image()로
긴 변을 LONG_SIDE(기본 448px)로 줄인 뒤 모델에 넣는다.
(Qwen2.5-VL은 28×28 패치 하나가 토큰 하나라서 해상도가 곧 속도와 메모리)
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from PIL import Image

from common import SYSTEM_PROMPT, USER_PROMPT

# mlx-community 4bit 변환본은 비양자화 가중치가 float16이라 학습 중 overflow → NaN (troubleshooting 참고).
# tools_convert_bf16.py로 bfloat16 변환한 로컬 사본이 있으면 그것을 기본 student로 쓴다.
STUDENT_HF = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
_LOCAL_BF16 = Path(__file__).resolve().parent / "models" / "qwen25vl-3b-4bit-bf16"
STUDENT = str(_LOCAL_BF16) if _LOCAL_BF16.exists() else STUDENT_HF
TEACHER = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
LONG_SIDE = 448

_TMP = Path(tempfile.gettempdir()) / "safestep_img"


def resize_long(img: Image.Image, long_side: int = LONG_SIDE) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = long_side / max(w, h)
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    return img


def prep_image(image, long_side: int = LONG_SIDE) -> str:
    """경로 또는 PIL 이미지 → 리사이즈된 임시 jpg 경로 (mlx-vlm은 파일 경로를 받음)."""
    _TMP.mkdir(parents=True, exist_ok=True)
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    img = resize_long(image, long_side)
    key = hashlib.md5(img.tobytes()).hexdigest()[:16]
    out = _TMP / f"{key}_{long_side}.jpg"
    if not out.exists():
        img.save(out, quality=92)
    return str(out)


def load(model_id: str = STUDENT, adapter: str | None = None):
    from mlx_vlm import load as _load

    return _load(model_id, adapter_path=adapter) if adapter else _load(model_id)


def messages(prompt: str, answer: str | None = None) -> list[dict]:
    m = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if answer is not None:
        m.append({"role": "assistant", "content": answer})
    return m


def generate(model, processor, image, prompt: str = USER_PROMPT, max_tokens: int = 256,
             temperature: float = 0.0, long_side: int = LONG_SIDE) -> str:
    from mlx_vlm import generate as _generate
    from mlx_vlm.prompt_utils import apply_chat_template

    text = apply_chat_template(processor, model.config, messages(prompt), num_images=1)
    out = _generate(model, processor, text, image=[prep_image(image, long_side)],
                    max_tokens=max_tokens, temperature=temperature, verbose=False)
    return (out.text if hasattr(out, "text") else str(out)).strip()


def peak_memory_gb() -> float:
    import mlx.core as mx

    return round(mx.get_peak_memory() / 1e9, 2)
