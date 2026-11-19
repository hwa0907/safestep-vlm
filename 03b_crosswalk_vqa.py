"""[3b단계] 횡단보도 VQA: Teacher에게 '바로 앞에 횡단보도가 있나?' 예/아니오만 묻는다.

배경: 테스트 100장 검수에서 수정 36건 중 15건이 횡단보도·차도 누락이었다.
AI Hub 바운딩박스 29종은 물체만 있고 노면 정보가 없기 때문.
7B 모델은 여러 조건을 섞은 규칙은 잘 못 따랐지만, 단순한 예/아니오 질문은 잘 답한다
→ 노면 정보만 VQA로 보충하고, 결과는 label_rules.apply_crosswalk()로 라벨에 반영한다.

사용 예:
    python 03b_crosswalk_vqa.py                          # 전체
    python 03b_crosswalk_vqa.py --eval data/reviewed.jsonl   # 검수 라벨과 비교해 정확도만 출력
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from backend import Runner, default_backend
from common import append_jsonl, read_jsonl

QUESTION = (
    "이 사진은 보행자 시점에서 촬영한 장면입니다. 보행자가 걷는 방향 바로 앞(몇 걸음 이내)에 "
    "건너야 할 횡단보도가 있습니까? 횡단보도가 옆 차도에만 보이거나 멀리 있으면 '아니오'입니다. "
    "'예' 또는 '아니오' 한 단어로만 답하세요."
)


QUESTION_CROP = (
    "이 사진은 보행자 발 앞 바닥을 확대한 것입니다. 바닥에 횡단보도의 굵은 흰색 줄무늬가 보입니까? "
    "'예' 또는 '아니오' 한 단어로만 답하세요."
)


def crop_ahead(path, x0=0.2, x1=0.8, y0=0.55):
    """진행 방향 바로 앞 영역(가운데, 화면 아래쪽)만 잘라낸다 — 옆/먼 곳의 횡단보도를 배제하는 ROI."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    return im.crop((int(w * x0), int(h * y0), int(w * x1), h))


def parse_yes(text: str) -> bool | None:
    s = text.strip().lstrip("'\"“ ")
    if s.startswith(("예", "네", "Yes", "yes")):
        return True
    if s.startswith(("아니", "No", "no")):
        return False
    return None


def evaluate(pred_file: str, reviewed_file: str):
    pred = {r["image"]: r["crosswalk"] for r in read_jsonl(pred_file)}
    tp = fp = fn = tn = 0
    for r in read_jsonl(reviewed_file):
        if r["status"] == "rejected" or r["image"] not in pred:
            continue
        gt = any(h["type"] == "횡단보도" and h["position"] == "가운데" for h in r["label"]["hazards"])
        p = bool(pred[r["image"]])
        tp += p and gt
        fp += p and not gt
        fn += (not p) and gt
        tn += (not p) and (not gt)
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    print(f"검수 라벨 {n}장 기준 — TP {tp} FP {fp} FN {fn} TN {tn} | "
          f"정밀도 {prec:.2f} 재현율 {rec:.2f} 정확도 {(tp + tn) / max(1, n):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="data/crosswalk.jsonl")
    ap.add_argument("--backend", default=default_backend(), choices=["mlx", "torch"])
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument("--only", nargs="*", default=None, help="이 jsonl들의 이미지만 (예: 검수한 테스트셋)")
    # 검수 100장 기준: full 정밀도 0.29 / crop 0.60 / both 0.75 (재현율 모두 1.00)
    ap.add_argument("--mode", default="both", choices=["both", "crop", "full"],
                    help="both: 발 앞 crop에서 '예'일 때만 전체 사진으로 재확인(기본) / crop / full")
    ap.add_argument("--eval", default="", help="검수 라벨 파일과 비교해 정확도 출력 후 종료")
    args = ap.parse_args()

    if args.eval:
        evaluate(args.out, args.eval)
        return

    items = read_jsonl(Path(args.frames) / "meta.jsonl")
    if args.only:
        allowed = {r["image"] for f in args.only for r in read_jsonl(f)}
        items = [r for r in items if r["image"] in allowed]
    done = {r["image"] for r in read_jsonl(args.out)}
    todo = [r for r in items if r["image"] not in done]
    print(f"대상 {len(items)} / 완료 {len(done)} / 이번 실행 {len(todo)}")

    runner = Runner(args.backend, role="teacher", long_side=args.long_side)
    n_yes, t0 = 0, time.time()
    for r in tqdm(todo, desc="crosswalk"):
        src = Path(args.frames) / r["image"]
        if args.mode == "full":
            raw = runner.generate(src, QUESTION, max_tokens=5)
            yes = parse_yes(raw)
        else:
            raw = runner.generate(crop_ahead(src), QUESTION_CROP, max_tokens=5)
            yes = parse_yes(raw)
            if args.mode == "both" and yes:  # 2단계 확인: 전체 사진에서도 '바로 앞 횡단보도'인가?
                raw2 = runner.generate(src, QUESTION, max_tokens=5)
                raw = f"crop={raw} | full={raw2}"
                yes = bool(parse_yes(raw2))
        n_yes += bool(yes)
        append_jsonl(args.out, {"image": r["image"], "crosswalk": bool(yes), "raw": raw,
                                "parsed": yes is not None})
    if todo:
        print(f"횡단보도 '예' {n_yes}/{len(todo)} · 평균 {(time.time() - t0) / len(todo):.2f}초/장")


if __name__ == "__main__":
    main()
