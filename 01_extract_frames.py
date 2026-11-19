"""[1단계] 보행 영상 → 프레임 추출 + 품질 필터링.

- 일정 간격(기본 1초)으로 프레임을 뽑는다.
- 흐린 프레임 제거: Laplacian 분산이 임계값보다 낮으면 버림 (걸으면서 찍으면 흔들림이 많음).
- 중복 제거: dHash(64bit) 해밍 거리가 가까우면 같은 장면으로 보고 버림 (신호 대기 등 정지 구간).
- (선택) 얼굴 블러: 공개 데이터셋으로 올릴 때 행인 얼굴 비식별화.

사용 예:
    python 01_extract_frames.py --videos data/raw_videos --out data/frames --interval 1.0 --blur-faces
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import write_jsonl

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(gray: np.ndarray, size: int = 8) -> int:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return int("".join("1" if b else "0" for b in diff.flatten()), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def blur_faces(img: np.ndarray, cascade) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    for (x, y, w, h) in faces:
        roi = img[y : y + h, x : x + w]
        img[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (0, 0), sigmaX=max(w, h) / 6)
    return len(faces)


def resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="영상 폴더")
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--interval", type=float, default=1.0, help="추출 간격(초)")
    ap.add_argument("--min-sharpness", type=float, default=60.0)
    ap.add_argument("--dup-threshold", type=int, default=6, help="dHash 해밍 거리 이하면 중복")
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--blur-faces", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cascade = None
    if args.blur_faces:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    videos = sorted(p for p in Path(args.videos).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"영상이 없습니다: {args.videos}")

    meta, stats = [], {"read": 0, "blurry": 0, "dup": 0, "kept": 0, "faces": 0}
    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * args.interval)))
        recent: list[int] = []  # 같은 영상 안에서 최근 해시만 비교
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            stats["read"] += 1
            frame = resize_long(frame, args.long_side)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if sharpness(gray) < args.min_sharpness:
                stats["blurry"] += 1
                idx += 1
                continue
            h = dhash(gray)
            if any(hamming(h, r) <= args.dup_threshold for r in recent[-10:]):
                stats["dup"] += 1
                idx += 1
                continue
            recent.append(h)
            if cascade is not None:
                stats["faces"] += blur_faces(frame, cascade)
            t = idx / fps
            name = f"{vp.stem}_{t:07.1f}.jpg"
            cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta.append({"image": name, "video": vp.stem, "time": round(t, 1)})
            stats["kept"] += 1
            idx += 1
        cap.release()
        print(f"[{vp.name}] 누적 저장 {stats['kept']}장")

    write_jsonl(out / "meta.jsonl", meta)
    print("통계:", stats)


if __name__ == "__main__":
    main()
