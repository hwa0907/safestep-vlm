"""GPU 없이 돌아가는 부분 테스트: 파싱, 지표, 프레임 추출, 영상 단위 분할.

    python tests/test_cpu.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import parse_output, read_jsonl, write_jsonl  # noqa: E402
from metrics import compute, rouge_l_char  # noqa: E402


def test_parse():
    raw = '설명입니다\n```json\n{"risk":"높음","hazards":[{"type":"전동킥보드","position":"왼쪽","distance":"가까움"},{"type":"우주선","position":"위","distance":"x"}],"guidance":"왼쪽에 킥보드가 있습니다."}\n```'
    p = parse_output(raw)
    assert p["risk"] == "높음"
    assert p["hazards"][1] == {"type": "기타", "position": "가운데", "distance": "보통"}
    assert parse_output("no json") is None
    assert parse_output('{"risk":"최고","hazards":[],"guidance":"x"}') is None
    assert parse_output('{"risk":"낮음","hazards":[],"guidance":""}') is None


def test_metrics():
    ref = {"risk": "높음", "hazards": [{"type": "사람", "position": "왼쪽", "distance": "가까움"},
                                      {"type": "볼라드", "position": "가운데", "distance": "보통"}],
           "guidance": "정면에 볼라드가 있습니다. 멈추세요."}
    assert compute([ref], [ref])["hazard_f1"] == 1.0
    assert compute([ref], [ref])["guidance_rougeL"] == 1.0
    pred = {"risk": "보통", "hazards": [{"type": "사람", "position": "오른쪽", "distance": "가까움"}],
            "guidance": "정면에 볼라드. 멈추세요."}
    m = compute([pred, None], [ref, ref])
    assert m["json_valid"] == 0.5 and m["risk_acc"] == 0.0
    assert m["hazard_p"] == 1.0 and m["hazard_r"] == 0.25
    assert m["position_acc"] == 0.0
    assert 0 < rouge_l_char(pred["guidance"], ref["guidance"]) < 1
    empty = {"risk": "낮음", "hazards": [], "guidance": "정면은 안전합니다."}
    m = compute([empty], [empty])
    assert m["hazard_f1"] == 1.0 and m["position_acc"] is None


def test_extract_and_split():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "videos").mkdir()
        rng = np.random.default_rng(0)
        for v in range(5):
            w = cv2.VideoWriter(str(d / "videos" / f"walk{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
            for i in range(60):  # 6초, 2초마다 장면이 바뀜 + 완전 흐린 구간 1개
                if i // 20 == 1 and v == 0:
                    frame = np.full((240, 320, 3), 128, np.uint8)
                else:
                    frame = (rng.random((240, 320, 3)) * 255).astype(np.uint8) if i % 20 == 0 else last
                last = frame
                w.write(frame)
            w.release()
        subprocess.run([sys.executable, str(ROOT / "01_extract_frames.py"), "--videos", str(d / "videos"),
                        "--out", str(d / "frames"), "--interval", "1.0"], check=True, cwd=ROOT)
        meta = read_jsonl(d / "frames" / "meta.jsonl")
        # 영상당 6프레임 중 장면 3개 → 중복 제거 후 최대 3장, 흐린 장면은 제거
        assert 0 < len(meta) <= 15, len(meta)
        assert all((d / "frames" / m["image"]).exists() for m in meta)

        label = {"risk": "낮음", "hazards": [], "guidance": "정면은 안전합니다."}
        rev = [{"image": m["image"], "video": m["video"], "status": "approved", "label": label} for m in meta]
        write_jsonl(d / "reviewed.jsonl", rev)
        subprocess.run([sys.executable, str(ROOT / "05_build_splits.py"), "--reviewed", str(d / "reviewed.jsonl"),
                        "--out", str(d / "splits"), "--ratios", "0.6,0.2,0.2"], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        vids = {s: {r["video"] for r in read_jsonl(d / "splits" / f"{s}.jsonl")} for s in ("train", "val", "test")}
        assert not (vids["train"] & vids["test"]) and not (vids["train"] & vids["val"]), vids
        assert sum(len(read_jsonl(d / "splits" / f"{s}.jsonl")) for s in vids) == len(meta)
        json.loads((d / "splits" / "stats.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
