"""[1단계 대안] 이미 프레임(이미지)으로 받은 공개 데이터셋 가져오기 (예: AI Hub 인도보행 영상).

영상을 직접 찍지 않고 다운로드한 이미지 폴더를 쓸 때 01_extract_frames.py 대신 실행한다.
- 하위 폴더(시퀀스/영상 단위)를 'video' 그룹으로 사용 → 05단계에서 그룹 단위 분할 (누수 방지)
- 시퀀스마다 --stride 장마다 1장만 사용 (연속 프레임은 거의 같은 장면)
- 01단계와 같은 흐림/중복 필터를 적용하고, 긴 변을 --long-side로 줄여 data/frames 에 복사

사용 예:
    python 01b_import_images.py --src /content/aihub/인도보행 --out data/frames --stride 10 --max-per-group 40 --max-total 1500
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import write_jsonl

import importlib.util

_spec = importlib.util.spec_from_file_location("ef", Path(__file__).with_name("01_extract_frames.py"))
ef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ef)

IMG_EXT = {".jpg", ".jpeg", ".png"}


def imread(path: Path):
    # 한글 경로에서 cv2.imread가 None을 돌려주는 문제 회피
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--group-depth", type=int, default=1,
                    help="이미지의 몇 단계 위 폴더를 그룹(영상)으로 볼지 (1=바로 위 폴더)")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-per-group", type=int, default=40)
    ap.add_argument("--max-total", type=int, default=1500)
    ap.add_argument("--min-sharpness", type=float, default=60.0)
    ap.add_argument("--dup-threshold", type=int, default=6)
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--blur-faces", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for p in src.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            g = p.parents[args.group_depth - 1]
            groups[str(g.relative_to(src)) if g != src else "root"].append(p)
    if not groups:
        raise SystemExit(f"이미지가 없습니다: {src}")

    names = sorted(groups)
    random.Random(args.seed).shuffle(names)  # 그룹 순서를 섞어 --max-total에 걸려도 다양하게
    cascade = None
    if args.blur_faces:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    meta, stats = [], {"groups": 0, "read": 0, "blurry": 0, "dup": 0, "kept": 0}
    for gi, g in enumerate(names):
        if stats["kept"] >= args.max_total:
            break
        files = sorted(groups[g])[:: args.stride]
        gid = f"g{gi:04d}"
        recent, kept_g = [], 0
        for f in files:
            if kept_g >= args.max_per_group or stats["kept"] >= args.max_total:
                break
            img = imread(f)
            if img is None:
                continue
            stats["read"] += 1
            img = ef.resize_long(img, args.long_side)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if ef.sharpness(gray) < args.min_sharpness:
                stats["blurry"] += 1
                continue
            h = ef.dhash(gray)
            if any(ef.hamming(h, r) <= args.dup_threshold for r in recent[-10:]):
                stats["dup"] += 1
                continue
            recent.append(h)
            if cascade is not None:
                ef.blur_faces(img, cascade)
            name = f"{gid}_{kept_g:04d}.jpg"
            cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta.append({"image": name, "video": gid, "source": str(f.relative_to(src)), "group": g})
            kept_g += 1
            stats["kept"] += 1
        stats["groups"] += kept_g > 0

    write_jsonl(out / "meta.jsonl", meta)
    print("통계:", stats)


if __name__ == "__main__":
    main()
