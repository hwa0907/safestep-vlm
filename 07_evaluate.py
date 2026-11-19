"""[7단계] 테스트셋 평가 + 모델 비교표.

비교 대상 (발표 결과 슬라이드):
  ① Student base (3B, zero-shot)         --name base3b
  ② Student + LoRA (3B, ours)            --name lora_r16 --adapter outputs/mlx-lora-r16
  ③ Teacher (7B, zero-shot, 힌트 없이)    --name teacher7b --role teacher
  (+ ablation: rank 8/16/32, vision LoRA 유무 등)

맥에서는 MLX, CUDA GPU에서는 PyTorch로 자동 선택.

사용 예:
    python 07_evaluate.py run --name base3b
    python 07_evaluate.py run --name lora_r16 --adapter outputs/mlx-lora-r16
    python 07_evaluate.py run --name teacher7b --role teacher
    python 07_evaluate.py table --names base3b lora_r16 teacher7b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend import Runner, default_backend
from common import USER_PROMPT, parse_output, read_jsonl, write_jsonl
from metrics import compute

COLS = ["json_valid", "risk_acc", "risk_macro_f1", "hazard_f1", "position_acc",
        "guidance_rougeL", "guidance_chrf", "latency_s", "peak_mem_gb"]


def run(args):
    rows = read_jsonl(args.test)
    if args.limit:
        rows = rows[: args.limit]
    runner = Runner(args.backend, role=args.role, model_id=args.model, adapter=args.adapter,
                    long_side=args.long_side, max_pixels=args.max_pixels, load_4bit=not args.no_4bit)

    preds, refs, records, times = [], [], [], []
    for r in rows:
        t0 = time.time()
        raw = runner.generate(Path(args.frames) / r["image"], USER_PROMPT)
        times.append(time.time() - t0)
        p = parse_output(raw)
        ref = json.loads(r["target"])
        preds.append(p)
        refs.append(ref)
        records.append({"image": r["image"], "raw": raw, "pred": p, "ref": ref})
        print(f"[{len(records)}/{len(rows)}] {r['image']} {'OK' if p else 'INVALID'} {times[-1]:.1f}s")

    m = compute(preds, refs)
    m["latency_s"] = round(sum(times) / len(times), 3)
    m["peak_mem_gb"] = runner.peak_memory_gb()
    m.update(backend=args.backend, model=runner.model_id, adapter=args.adapter)
    out = Path(args.out_dir)
    write_jsonl(out / f"{args.name}_pred.jsonl", records)
    (out / f"{args.name}_metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=2))


def table(args):
    out = Path(args.out_dir)
    lines = ["| model | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for n in args.names:
        m = json.loads((out / f"{n}_metrics.json").read_text(encoding="utf-8"))
        lines.append(f"| {n} | " + " | ".join(str(m.get(c, "-")) for c in COLS) + " |")
    md = "\n".join(lines)
    (out / "comparison.md").write_text(md + "\n", encoding="utf-8")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--backend", default=default_backend(), choices=["mlx", "torch"])
    r.add_argument("--role", default="student", choices=["student", "teacher"])
    r.add_argument("--model", default=None)
    r.add_argument("--adapter", default=None)
    r.add_argument("--test", default="data/splits/test.jsonl")
    r.add_argument("--frames", default="data/frames")
    r.add_argument("--long-side", type=int, default=None, help="[mlx] 기본 448 (학습과 동일)")
    r.add_argument("--max-pixels", type=int, default=448 * 448, help="[torch] 학습 때와 같은 값")
    r.add_argument("--no-4bit", action="store_true")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out-dir", default="outputs/eval")
    t = sub.add_parser("table")
    t.add_argument("--names", nargs="+", required=True)
    t.add_argument("--out-dir", default="outputs/eval")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else table(args)


if __name__ == "__main__":
    main()
