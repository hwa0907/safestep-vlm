"""mlx-community 4bit 모델의 float16 가중치(비양자화 층·양자화 scale/bias)를 bfloat16으로 변환해 로컬에 저장.

이유: 학습 150 iteration에서 loss가 갑자기 NaN. config는 bfloat16인데 실제 텐서는 F16이라
float16 범위(최대 약 65,504)를 넘는 활성값/기울기가 inf → NaN이 됨. bfloat16은 float32와 같은 지수 범위.

    python tools_convert_bf16.py mlx-community/Qwen2.5-VL-3B-Instruct-4bit models/qwen25vl-3b-4bit-bf16
"""
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

src = Path(snapshot_download(sys.argv[1]))
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
n = 0
for f in src.iterdir():
    if f.suffix == ".safetensors":
        w = mx.load(str(f))
        out = {}
        for k, v in w.items():
            if v.dtype == mx.float16:
                v = v.astype(mx.bfloat16)
                n += 1
            out[k] = v
        mx.save_safetensors(str(dst / f.name), out, metadata={"format": "mlx"})
    elif f.is_file():
        shutil.copy(f, dst / f.name)
print(f"F16 → BF16 변환 텐서 {n}개 → {dst}")
