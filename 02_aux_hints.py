"""[2단계] Teacher용 보조 힌트: 물체 박스 + 단안 깊이 추정(Depth Anything V2).

박스 출처 두 가지
  - --gt-src 지정 시: AI Hub 인도보행 데이터의 **사람이 단 정답 박스**(CVAT XML, 29종) ← 권장
  - 없으면: YOLO11 자동 검출 (직접 촬영 영상 등 라벨 없는 데이터용)

각 박스에 대해
  - 위치: 박스 중심 x가 화면 1/3 기준 왼쪽/가운데/오른쪽
  - 거리: 박스 안 상대 깊이(0~255, 클수록 가까움)의 중앙값 → 가까움/보통/멀리
를 계산해 teacher 프롬프트에 붙일 텍스트를 만든다 (수업 5주차 검출 + 10주차 깊이 추정 실습의 연장).

사용 예:
    python 02_aux_hints.py --gt-src ~/aihub/extracted      # AI Hub 정답 박스 사용
    python 02_aux_hints.py                                 # YOLO 사용
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import read_jsonl, write_jsonl

# YOLO(COCO) 클래스 → 위험요소 어휘
COCO2KO = {
    "person": "사람", "bicycle": "자전거", "motorcycle": "오토바이",
    "car": "자동차", "truck": "자동차", "bus": "자동차",
    "traffic light": "신호등", "fire hydrant": "기타", "stop sign": "기타",
    "bench": "기타", "dog": "동물", "cat": "동물", "potted plant": "기타",
    "suitcase": "기타", "chair": "기타",
}
# AI Hub 인도보행 29종 → (위험요소 어휘, 힌트에 쓸 세부 이름)
AIHUB2KO = {
    "person": ("사람", "사람"), "bicycle": ("자전거", "자전거"), "scooter": ("전동킥보드", "킥보드"),
    "motorcycle": ("오토바이", "오토바이"), "car": ("자동차", "승용차"), "bus": ("자동차", "버스"),
    "truck": ("자동차", "트럭"), "carrier": ("기타", "손수레"), "stroller": ("기타", "유모차"),
    "wheelchair": ("기타", "휠체어"), "dog": ("동물", "개"), "cat": ("동물", "고양이"),
    "movable_signage": ("입간판", "이동식 안내판"), "bollard": ("볼라드", "볼라드"),
    "pole": ("기둥", "기둥"), "tree_trunk": ("가로수", "가로수"), "barricade": ("바리케이드", "바리케이드"),
    "bench": ("기타", "벤치"), "chair": ("기타", "의자"), "table": ("기타", "탁자"),
    "potted_plant": ("기타", "화분"), "fire_hydrant": ("기타", "소화전"), "kiosk": ("기타", "가판대/키오스크"),
    "parking_meter": ("기타", "주차정산기"), "power_controller": ("기타", "전력제어함"),
    "traffic_light_controller": ("기타", "신호제어기"), "stop": ("기타", "정류장"),
    "traffic_light": ("신호등", "신호등"), "traffic_sign": ("기타", "교통표지판"),
}
# 보행 동선과 거의 무관하고 화면 위쪽에 있는 경우가 많은 것 (멀리 있을 때 힌트에서 제외)
LOW_PRIORITY = {"traffic_sign", "traffic_light"}


def position(cx: float, w: int) -> str:
    return "왼쪽" if cx < w / 3 else ("오른쪽" if cx > 2 * w / 3 else "가운데")


def distance(d: float, near: float, far: float) -> str:
    return "가까움" if d >= near else ("보통" if d >= far else "멀리")


def load_cvat(src: Path) -> dict[str, list]:
    """CVAT XML들 → {"폴더/파일명.jpg": [(label, xtl, ytl, xbr, ybr, xml_w, xml_h), ...]}"""
    boxes = {}
    for xml in src.rglob("*.xml"):
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            print("XML 파싱 실패(건너뜀):", xml)
            continue
        folder = xml.parent.relative_to(src)
        for im in root.iter("image"):
            w, h = float(im.get("width")), float(im.get("height"))
            key = str(folder / im.get("name"))
            boxes[key] = [(b.get("label"), float(b.get("xtl")), float(b.get("ytl")),
                           float(b.get("xbr")), float(b.get("ybr")), w, h) for b in im.iter("box")]
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--out", default="data/hints.jsonl")
    ap.add_argument("--gt-src", default="", help="AI Hub 압축 해제 폴더 (CVAT XML 정답 박스 사용)")
    ap.add_argument("--yolo", default="yolo11n.pt")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-area", type=float, default=0.002, help="이미지 대비 박스 면적 하한 (아주 작은 것 제외)")
    # 기본값은 AI Hub 1,452장 객체 7,834개의 깊이 분포(p50=60, p70=88, p85=130)와 육안 확인으로 보정
    ap.add_argument("--near", type=float, default=140, help="상대깊이(0-255) 이상이면 가까움")
    ap.add_argument("--far", type=float, default=70, help="이 값 미만이면 멀리")
    ap.add_argument("--bottom-near", type=float, default=0.95,
                    help="박스 아래쪽이 화면 높이의 이 비율 이상(화면 하단에 닿음)이면 가까움")
    ap.add_argument("--max-objects", type=int, default=8)
    args = ap.parse_args()

    from transformers import pipeline

    depth = pipeline("depth-estimation", model=args.depth_model)
    gt = load_cvat(Path(args.gt_src).expanduser()) if args.gt_src else None
    yolo = None
    if gt is None:
        from ultralytics import YOLO

        yolo = YOLO(args.yolo)
    else:
        print(f"정답 박스 로드: 이미지 {len(gt)}장")

    frames = Path(args.frames)
    rows, n_gt, n_miss = [], 0, 0
    for m in tqdm(read_jsonl(frames / "meta.jsonl"), desc="hints"):
        img = Image.open(frames / m["image"]).convert("RGB")
        w, h = img.size
        dmap = np.array(depth(img)["depth"].resize((w, h)), dtype=np.float32)

        cand = []  # (type, name, label, x1, y1, x2, y2)
        if gt is not None and m.get("source") in gt:
            n_gt += 1
            for label, x1, y1, x2, y2, xw, xh in gt[m["source"]]:
                if label not in AIHUB2KO:
                    continue
                sx, sy = w / xw, h / xh
                t, name = AIHUB2KO[label]
                cand.append((t, name, label, x1 * sx, y1 * sy, x2 * sx, y2 * sy))
        elif yolo is not None:
            res = yolo.predict(img, conf=args.conf, verbose=False)[0]
            for box, cls in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
                name = res.names[int(cls)]
                if name in COCO2KO:
                    cand.append((COCO2KO[name], COCO2KO[name], name, *box))
        else:
            n_miss += 1

        objs = []
        for t, name, label, x1, y1, x2, y2 in cand:
            if (x2 - x1) * (y2 - y1) < args.min_area * w * h:
                continue
            xi1, yi1, xi2, yi2 = map(int, (x1, y1, x2, y2))
            patch = dmap[max(0, yi1): max(yi1 + 1, yi2), max(0, xi1): max(xi1 + 1, xi2)]
            d = float(np.median(patch)) if patch.size else 0.0
            dist = distance(d, args.near, args.far)
            if y2 >= args.bottom_near * h:  # 지면 접점이 화면 맨 아래 = 발 앞 (핀홀 카메라에서 아래일수록 가까움)
                dist = "가까움"
            if label in LOW_PRIORITY and dist == "멀리":
                continue
            objs.append({"type": t, "name": name, "label": label,
                         "position": position((x1 + x2) / 2, w), "distance": dist, "depth": round(d, 1),
                         "box": [round(v, 1) for v in (x1, y1, x2, y2)]})
        objs.sort(key=lambda o: -o["depth"])  # 가까운 것부터
        objs = objs[: args.max_objects]
        hint_text = ", ".join(f"{o['name']}({o['position']}, {o['distance']})" for o in objs) or "주요 물체 없음"
        rows.append({"image": m["image"], "video": m["video"], "source": m.get("source"),
                     "box_source": "gt" if gt is not None else "yolo", "objects": objs, "hint_text": hint_text})

    write_jsonl(args.out, rows)
    msg = f"{len(rows)}장 힌트 저장 → {args.out}"
    if gt is not None:
        msg += f" (정답 박스 매칭 {n_gt}장, 매칭 실패 {n_miss}장)"
    print(msg)


if __name__ == "__main__":
    main()
