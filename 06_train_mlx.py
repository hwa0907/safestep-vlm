"""[6단계 - 맥] Student(Qwen2.5-VL-3B, 4bit) QLoRA 학습 — Apple Silicon / MLX.

하는 일
1) data/splits/{train,val}.jsonl → mlx-vlm 학습 형식으로 변환
   - 이미지를 긴 변 448px로 줄여 work/images/ 에 저장 (추론 때와 같은 해상도)
   - {"images": ["images/xxx.jpg"], "messages": [system, user(프롬프트), assistant(정답 JSON)]}
2) mlx_vlm.train 을 LoRA + --train-on-completions 로 실행
   - base 가중치는 4bit로 고정, 언어모델 선형층에만 LoRA (W + (α/r)·BA)
   - completion-only loss: 프롬프트·이미지 토큰은 빼고 답변(JSON) 토큰에만 loss
3) 로그에서 train/val loss를 뽑아 loss_curve.png 저장

사용 예:
    python 06_train_mlx.py --out outputs/mlx-lora-r16
    python 06_train_mlx.py --rank 8 --alpha 16 --out outputs/mlx-lora-r8        # ablation
    python 06_train_mlx.py --train-vision --out outputs/mlx-lora-r16-vis        # ablation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

from common import USER_PROMPT, read_jsonl, write_jsonl
from mlx_backend import LONG_SIDE, STUDENT, messages, resize_long


def build(split_file: Path, frames: Path, work: Path, name: str, long_side: int) -> int:
    rows = read_jsonl(split_file)
    img_dir = work / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        dst = img_dir / r["image"]
        if not dst.exists():
            resize_long(Image.open(frames / r["image"]), long_side).save(dst, quality=92)
        out.append({"images": [f"images/{r['image']}"], "messages": messages(USER_PROMPT, r["target"])})
    write_jsonl(work / f"{name}.jsonl", out)
    return len(out)


def parse_losses(log: str):
    """mlx-vlm 로그 형식:
        Training (iteration 12):
        optimizer_step=6, train_loss=0.0056, ...
        Validation (iteration 12):
        val_loss=0.1467, ...
    """
    tr, va, step = [], [], None
    for line in log.splitlines():
        h = re.search(r"\(iteration (\d+)\)", line)
        if h:
            step = int(h.group(1))
        if step is None:
            continue
        m = re.search(r"\btrain_loss=([0-9.eE+-]+)", line)
        if m:
            tr.append((step, float(m.group(1))))
        m = re.search(r"\bval_loss=([0-9.eE+-]+)", line)
        if m:
            va.append((step, float(m.group(1))))
    return tr, va


def plot(tr, va, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    if tr:
        plt.plot(*zip(*tr), label="train loss")
    if va:
        plt.plot(*zip(*va), "o-", label="val loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--work", default="data/mlx_sft")
    ap.add_argument("--out", default="outputs/mlx-lora-r16")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--iters", type=int, default=0, help="0이면 epochs로 계산")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--long-side", type=int, default=LONG_SIDE)
    ap.add_argument("--train-vision", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits, frames, work, out = Path(args.splits), Path(args.frames), Path(args.work), Path(args.out)
    n_tr = build(splits / "train.jsonl", frames, work, "train", args.long_side)
    n_va = build(splits / "val.jsonl", frames, work, "validation", args.long_side)
    print(f"train {n_tr} / val {n_va} → {work}")

    # iteration = 미니배치 1개. 1 epoch = n_tr / batch iterations
    iters = args.iters or math.ceil(n_tr / args.batch) * args.epochs
    eval_every = max(10, math.ceil(n_tr / args.batch) // 2)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "mlx_vlm.train",
           "--model", args.model, "--data", str(work),
           "--train-type", "lora", "--lora-rank", str(args.rank), "--lora-alpha", str(args.alpha),
           "--lora-dropout", str(args.dropout), "--learning-rate", str(args.lr),
           "--iters", str(iters), "--batch-size", str(args.batch),
           "--gradient-accumulation-steps", str(args.grad_accum),
           "--train-on-completions", "--grad-checkpoint",
           "--steps-per-report", "5", "--steps-per-eval", str(eval_every),
           "--steps-per-save", str(eval_every), "--val-batches", "-1",
           "--seed", str(args.seed), "--output-path", str(out)]
    if args.train_vision:
        cmd.append("--train-vision")
    print(" ".join(cmd))

    log_lines = []
    # Anaconda 파이썬에서는 MLX가 Anaconda의 MPICH(libmpi)를 찾아 분산 통신 중 abort(-6)한다.
    # 단일 맥 학습이므로 MPI 라이브러리 이름을 없는 이름으로 바꿔 MPI를 끈다.
    env = {**os.environ, "MLX_MPI_LIBNAME": "libmpi_disabled.dylib"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    for line in proc.stdout:
        print(line, end="")
        log_lines.append(line)
        if "train_loss=nan" in line or "val_loss=nan" in line:  # 발산하면 몇 시간 낭비하지 말고 즉시 중단
            proc.kill()
            log_lines.append("\n[NaN 감지 → 학습 중단]\n")
            break
    code = proc.wait()
    log = "".join(log_lines)
    (out / "train_log.txt").write_text(log, encoding="utf-8")
    (out / "train_args.json").write_text(json.dumps({**vars(args), "iters": iters}, indent=2), encoding="utf-8")
    if code != 0:
        raise SystemExit(f"학습 실패 (종료 코드 {code}). {out / 'train_log.txt'} 확인")
    tr, va = parse_losses(log)
    if tr or va:
        plot(tr, va, out / "loss_curve.png")
    print(f"완료 → 어댑터: {out}  (loss 기록 train {len(tr)}개, val {len(va)}개)")


if __name__ == "__main__":
    main()
