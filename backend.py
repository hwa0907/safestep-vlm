"""추론 백엔드 선택: mlx(맥, 기본) / torch(CUDA GPU).

    r = Runner("mlx", role="student", adapter="outputs/mlx-lora-r16")
    text = r.generate(image, prompt, max_tokens=256, temperature=0.0)
"""
from __future__ import annotations

import platform

from common import USER_PROMPT

TORCH_IDS = {"student": "Qwen/Qwen2.5-VL-3B-Instruct", "teacher": "Qwen/Qwen2.5-VL-7B-Instruct"}


def default_backend() -> str:
    return "mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "torch"


class Runner:
    def __init__(self, backend: str, role: str = "student", model_id: str | None = None,
                 adapter: str | None = None, long_side: int | None = None,
                 max_pixels: int = 448 * 448, load_4bit: bool = True):
        self.backend = backend
        if backend == "mlx":
            import mlx_backend as mb

            self.model_id = model_id or (mb.TEACHER if role == "teacher" else mb.STUDENT)
            self.long_side = long_side or mb.LONG_SIDE
            self.model, self.processor = mb.load(self.model_id, adapter)
            self._mb = mb
        elif backend == "torch":
            from common import load_model, load_processor

            self.model_id = model_id or TORCH_IDS[role]
            self.processor = load_processor(self.model_id, max_pixels)
            self.model = load_model(self.model_id, load_4bit=load_4bit, adapter=adapter)
        else:
            raise ValueError(backend)
        self.adapter = adapter

    def generate(self, image, prompt: str = USER_PROMPT, max_tokens: int = 256, temperature: float = 0.0) -> str:
        if self.backend == "mlx":
            return self._mb.generate(self.model, self.processor, image, prompt, max_tokens=max_tokens,
                                     temperature=temperature, long_side=self.long_side)
        return _torch_generate(self.model, self.processor, image, prompt, max_tokens, temperature)

    def peak_memory_gb(self) -> float | None:
        if self.backend == "mlx":
            return self._mb.peak_memory_gb()
        import torch

        return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


def _torch_generate(model, processor, image, prompt, max_tokens, temperature):
    import torch
    from PIL import Image

    from common import build_messages

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    text = processor.apply_chat_template(build_messages(prompt), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_tokens, do_sample=temperature > 0)
    if temperature > 0:
        kw.update(temperature=temperature, top_p=0.9)
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
