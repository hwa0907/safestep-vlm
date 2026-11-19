"""[3단계] Teacher VLM으로 pseudo label 생성 (수업 4주차: Knowledge distillation / Self-training).

Teacher = Qwen2.5-VL-7B-Instruct (4bit, 이미지 + YOLO/깊이 힌트)
Student = Qwen2.5-VL-3B-Instruct (4bit, 이미지만) ← 6단계에서 LoRA로 학습

- 맥(Apple Silicon)에서는 MLX, CUDA GPU에서는 PyTorch로 자동 선택 (--backend 로 지정 가능).
- Teacher는 student(448px)보다 큰 768px 이미지를 본다 → 더 좋은 라벨.
- 중간에 끊겨도 이어서 돌 수 있게 이미 라벨링한 이미지는 건너뛴다.
- JSON 형식이 깨지면 최대 --retries 번 재시도 (온도 0.7로 샘플링).

사용 예:
    python 03_teacher_label.py --limit 20      # 먼저 20장만 돌려 품질·속도 확인
    python 03_teacher_label.py                 # 전체
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from backend import Runner, default_backend
from common import append_jsonl, normalize_label, parse_output, read_jsonl, teacher_prompt
from label_rules import (clean_guidance, decide_action, guidance_prompt, risk_level, select_hazards,
                         template_guidance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--hints", default="data/hints.jsonl", help="없으면 힌트 없이 라벨링")
    ap.add_argument("--out", default="data/teacher.jsonl")
    ap.add_argument("--backend", default=default_backend(), choices=["mlx", "torch"])
    ap.add_argument("--model", default=None, help="기본: 백엔드별 Qwen2.5-VL-7B")
    ap.add_argument("--long-side", type=int, default=768, help="[mlx] teacher 입력 해상도")
    ap.add_argument("--max-pixels", type=int, default=768 * 28 * 28, help="[torch]")
    ap.add_argument("--no-4bit", action="store_true", help="[torch]")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "full"],
                    help="hybrid: 정답 박스로 risk/hazards 규칙 생성 + Teacher는 안내문만 (정답 박스 있을 때) / "
                         "full: Teacher가 JSON 전체 생성")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="--limit 개수를 무작위로 고름 (파일럿용)")
    ap.add_argument("--priority", default="", help="split_assign.json — test/val 시퀀스를 먼저 라벨링")
    args = ap.parse_args()

    frames = Path(args.frames)
    hints = {r["image"]: r for r in read_jsonl(args.hints)}
    items = list(hints.values()) if hints else read_jsonl(frames / "meta.jsonl")
    done = {r["image"] for r in read_jsonl(args.out)}
    todo = [r for r in items if r["image"] not in done]
    if args.priority:  # 예: 검수할 val/test 이미지를 먼저 라벨링
        import json as _json

        assign = _json.loads(Path(args.priority).read_text(encoding="utf-8"))
        rank = {"test": 0, "val": 1, "train": 2}
        todo.sort(key=lambda r: rank.get(assign.get(r.get("video"), "train"), 2))
        print("우선순위 적용: test → val → train")
    if args.limit:
        if args.sample:  # 파일럿: 여러 시퀀스에서 고르게 무작위 추출
            import random

            todo = random.Random(0).sample(todo, min(args.limit, len(todo)))
        else:
            todo = todo[: args.limit]
    print(f"전체 {len(items)} / 완료 {len(done)} / 이번 실행 {len(todo)} · backend={args.backend}"
          f" · 힌트 {'사용' if hints else '없음'}")

    runner = Runner(args.backend, role="teacher", model_id=args.model, long_side=args.long_side,
                    max_pixels=args.max_pixels, load_4bit=not args.no_4bit)

    n_ok, n_teacher, t0 = 0, 0, time.time()
    for r in tqdm(todo, desc="teacher"):
        hybrid = args.mode == "hybrid" and r.get("box_source") == "gt"
        raw, label = "", None
        source = "full"
        if hybrid:
            # 위험도·위험요소·행동은 정답 박스+깊이로 규칙 생성 → Teacher는 사진을 보고 문장만 다듬음.
            # Teacher 문장이 방향/물체/행동을 바꾸면 탈락시키고 규칙 문장(template)을 그대로 쓴다.
            objects = r.get("objects", [])
            hazards = select_hazards(objects)
            risk = risk_level(hazards)
            action = decide_action(risk, hazards, objects)
            template = template_guidance(hazards, action)
            prompt = guidance_prompt(template)
            g = None
            for attempt in range(args.retries + 1):
                raw = runner.generate(frames / r["image"], prompt, max_tokens=120,
                                      temperature=0.7 if attempt else 0.0)
                g = clean_guidance(raw, hazards, action)
                if g:
                    break
            source = "teacher" if g else "template"
            label = normalize_label({"risk": risk, "hazards": hazards, "guidance": g or template})
            n_teacher += source == "teacher"
        else:
            prompt = teacher_prompt(r.get("hint_text"))
            for attempt in range(args.retries + 1):
                raw = runner.generate(frames / r["image"], prompt, max_tokens=320,
                                      temperature=0.7 if attempt else 0.0)
                label = parse_output(raw)
                if label:
                    break
        n_ok += label is not None
        append_jsonl(args.out, {
            "image": r["image"], "video": r.get("video"), "hint_text": r.get("hint_text"),
            "mode": "hybrid" if hybrid else "full", "guidance_source": source,
            "raw": raw, "label": label, "ok": label is not None,
        })
    if todo:
        if args.mode == "hybrid":
            print(f"안내문: Teacher 문장 채택 {n_teacher} / 규칙 문장 대체 {n_ok - n_teacher}")
        print(f"JSON 유효 {n_ok}/{len(todo)} · 평균 {(time.time() - t0) / len(todo):.1f}초/장"
              f" · 최대 메모리 {runner.peak_memory_gb()} GB")


if __name__ == "__main__":
    main()
