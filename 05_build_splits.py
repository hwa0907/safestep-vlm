"""[5단계] 학습/검증/테스트 분할 + 데이터셋 통계 + (선택) HuggingFace Hub 업로드.

중요: 프레임 단위로 무작위 분할하면 같은 영상의 거의 같은 장면이 train/test에 동시에 들어가
성능이 부풀려진다(data leakage). → **영상(video) 단위로** 분할한다.

사용 예:
    python 05_build_splits.py --reviewed data/reviewed.jsonl --out data/splits
    python 05_build_splits.py ... --push-to-hub <hf_user>/safestep-ko --private
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import dumps, read_jsonl, write_jsonl


def vid(r):
    return r.get("video") or r["image"].rsplit("_", 1)[0]


def assign_videos(rows, ratios, seed) -> dict[str, str]:
    by_video = defaultdict(int)
    for r in rows:
        by_video[vid(r)] += 1
    vids = sorted(by_video)
    random.Random(seed).shuffle(vids)
    total = len(rows)
    count = {"train": 0, "val": 0, "test": 0}
    targets = {"test": ratios[2] * total, "val": ratios[1] * total}
    assign = {}
    for v in vids:  # 작은 split부터 채운다
        s = "test" if count["test"] < targets["test"] else ("val" if count["val"] < targets["val"] else "train")
        assign[v] = s
        count[s] += by_video[v]
    return assign


def group_split(rows, ratios, seed, assign_file: Path):
    """영상(시퀀스) 단위 분할. 처음 정한 배정을 파일로 저장해 두고 이후 실행에서도 그대로 쓴다
    (검수에서 일부가 폐기돼 개수가 바뀌어도 시퀀스가 train↔test로 옮겨 다니지 않게)."""
    if assign_file.exists():
        assign = json.loads(assign_file.read_text(encoding="utf-8"))
    else:
        assign = assign_videos(rows, ratios, seed)
        assign_file.parent.mkdir(parents=True, exist_ok=True)
        assign_file.write_text(json.dumps(assign, ensure_ascii=False, indent=1), encoding="utf-8")
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        splits[assign.get(vid(r), "train")].append(r)
    return splits


def describe(rows):
    risk = Counter(r["label"]["risk"] for r in rows)
    haz = Counter(h["type"] for r in rows for h in r["label"]["hazards"])
    empty = sum(1 for r in rows if not r["label"]["hazards"])
    glen = sum(len(r["label"]["guidance"]) for r in rows) / max(1, len(rows))
    return {"n": len(rows), "risk": dict(risk), "hazard_types": dict(haz.most_common()),
            "no_hazard": empty, "avg_guidance_chars": round(glen, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewed", default="data/reviewed.jsonl")
    ap.add_argument("--teacher", default="data/teacher.jsonl")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="검수 안 한 teacher 라벨도 사용 (검수 전 빠른 실험용)")
    ap.add_argument("--crosswalk", default="", help="03b 횡단보도 VQA 결과(data/crosswalk.jsonl)")
    ap.add_argument("--test-reviewed-only", action="store_true",
                    help="test에는 사람이 검수(승인·수정)한 라벨만 남김")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--ratios", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--push-to-hub", default="")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--include-images", action="store_true",
                    help="원본 이미지도 업로드 (직접 촬영 데이터만! AI Hub 데이터는 재배포 금지)")
    args = ap.parse_args()

    reviewed = [{**r, "label_source": "reviewed", "reviewer": r.get("reviewer", "human")} for r in read_jsonl(args.reviewed)
                if r["status"] in ("approved", "edited")]
    rows = reviewed
    if args.allow_unreviewed:
        seen = {r["image"] for r in read_jsonl(args.reviewed)}  # 폐기(rejected)된 것도 제외
        rows = reviewed + [{**r, "label_source": "teacher"} for r in read_jsonl(args.teacher)
                           if r["ok"] and r["image"] not in seen]
    if not rows:
        raise SystemExit("사용할 라벨이 없습니다. 4단계 검수를 하거나 --allow-unreviewed를 쓰세요.")
    n_cw = 0
    if args.crosswalk:  # 횡단보도 VQA 결과를 자동 라벨에만 반영 (검수 라벨은 그대로)
        from label_rules import apply_crosswalk

        cw = {r["image"] for r in read_jsonl(args.crosswalk) if r["crosswalk"]}
        for r in rows:
            if r["label_source"] == "teacher" and r["image"] in cw:
                r["label"] = apply_crosswalk(r["label"])
                n_cw += 1

    ratios = [float(x) for x in args.ratios.split(",")]
    out = Path(args.out)
    splits = group_split(rows, ratios, args.seed, out / "split_assign.json")
    if args.test_reviewed_only:  # 최종 점수를 내는 test는 검수(승인·수정)된 라벨만 (val은 loss 확인용이라 teacher 라벨 사용)
        splits["test"] = [r for r in splits["test"] if r["label_source"] == "reviewed"]
    stats = {"crosswalk_applied": n_cw}
    for name, rs in splits.items():
        write_jsonl(out / f"{name}.jsonl",
                    [{"image": r["image"], "video": r.get("video"), "target": dumps(r["label"]),
                      "label_source": r["label_source"]} for r in rs])
        stats[name] = {**describe(rs), "label_source": dict(Counter(r["label_source"] for r in rs))}

    allrev = read_jsonl(args.reviewed)
    if allrev:
        c = Counter(r["status"] for r in allrev)
        stats["review"] = {**c, "approve_rate": round(c["approved"] / len(allrev), 3),
                           "edit_rate": round(c["edited"] / len(allrev), 3),
                           "reject_rate": round(c["rejected"] / len(allrev), 3)}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.push_to_hub:
        from datasets import Dataset, DatasetDict, Image

        # AI Hub 약관(제3자 제공 금지) → 기본은 원본 이미지 없이 라벨 + 원본 파일 경로만 업로드.
        # 직접 촬영한 이미지처럼 공개해도 되는 경우에만 --include-images.
        meta = {m["image"]: m for m in read_jsonl(Path(args.frames) / "meta.jsonl")}
        dd = {}
        for name in splits:
            rs = read_jsonl(out / f"{name}.jsonl")
            cols = {
                "image_name": [r["image"] for r in rs],
                "source_file": [meta.get(r["image"], {}).get("source", "") for r in rs],
                "video": [r["video"] for r in rs],
                "target": [r["target"] for r in rs],
            }
            if args.include_images:
                cols["image"] = [str(Path(args.frames) / r["image"]) for r in rs]
                dd[name] = Dataset.from_dict(cols).cast_column("image", Image())
            else:
                dd[name] = Dataset.from_dict(cols)
        DatasetDict(dd).push_to_hub(args.push_to_hub, private=args.private)
        print(f"업로드 완료: https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
