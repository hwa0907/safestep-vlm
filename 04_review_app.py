"""[4단계] 사람 검수 도구 (Gradio).

Teacher 라벨을 한 장씩 보며 승인 / 수정 후 저장 / 폐기한다.
- 사진 위에 정답 박스를 거리별 색으로 그려 보여준다 (빨강=가까움, 노랑=보통, 하늘=멀리).
- --only 로 검수할 이미지를 제한할 수 있다 (예: 평가에 쓰는 val/test만 사람이 검수).
- 검수 결과는 data/reviewed.jsonl 에 누적되고, 다시 켜면 안 본 것부터 이어서 보여준다.
- 검수 통계(승인율, 수정율)는 발표의 "데이터셋 품질" 근거로 쓴다.

사용 예:
    python 04_review_app.py --only data/splits/test.jsonl data/splits/val.jsonl
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from common import append_jsonl, parse_output, read_jsonl

COLORS = {"가까움": (255, 40, 40), "보통": (255, 210, 0), "멀리": (0, 200, 255)}
RENDER_DIR = Path(tempfile.gettempdir()) / "safestep_review"


def render(frames: Path, image: str, objects: list[dict]) -> str:
    """정답 박스를 그린 미리보기 이미지를 임시 폴더에 저장하고 경로를 돌려준다."""
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    out = RENDER_DIR / image
    if not out.exists():
        im = Image.open(frames / image).convert("RGB")
        d = ImageDraw.Draw(im)
        for o in objects:
            if "box" not in o:
                continue
            c = COLORS.get(o["distance"], (255, 255, 255))
            d.rectangle(o["box"], outline=c, width=4)
            d.text((o["box"][0] + 4, o["box"][1] + 2), o.get("name", o["type"]), fill=c)
        im.save(out, quality=90)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--teacher", default="data/teacher.jsonl")
    ap.add_argument("--hints", default="data/hints.jsonl", help="박스 그리기용 (없으면 사진만)")
    ap.add_argument("--only", nargs="*", default=None, help="이 jsonl들에 있는 이미지만 검수")
    ap.add_argument("--shuffle", action="store_true", help="무작위 순서로 검수 (일부만 해도 대표 표본)")
    ap.add_argument("--out", default="data/reviewed.jsonl")
    ap.add_argument("--share", action="store_true", help="Colab에서 공개 링크로 열기")
    args = ap.parse_args()

    frames = Path(args.frames)
    teacher = read_jsonl(args.teacher)
    hints = {r["image"]: r.get("objects", []) for r in read_jsonl(args.hints)}
    if args.only:
        allowed = {r["image"] for f in args.only for r in read_jsonl(f)}
        teacher = [r for r in teacher if r["image"] in allowed]
    if args.shuffle:  # 중간에 멈춰도 검수한 부분이 전체에서 고르게 뽑힌 표본이 되도록
        import random

        random.Random(0).shuffle(teacher)
    done = {r["image"] for r in read_jsonl(args.out)}
    queue = [r for r in teacher if r["image"] not in done]
    print(f"검수 대상 {len(teacher)}장 / 남은 것 {len(queue)}장")

    def pretty(r):
        if r.get("label"):
            return json.dumps(r["label"], ensure_ascii=False, indent=2)
        return r.get("raw", "")

    def stats():
        c = Counter(r["status"] for r in read_jsonl(args.out))
        return " / ".join(f"{k} {v}" for k, v in c.items()) or "아직 없음"

    def show(i):
        if i >= len(queue):
            return None, "", "", f"검수 완료! ({stats()})"
        r = queue[i]
        src = f" · 안내문 출처: {r['guidance_source']}" if r.get("guidance_source") else ""
        return (render(frames, r["image"], hints.get(r["image"], [])), pretty(r), r.get("hint_text") or "",
                f"**{i + 1}/{len(queue)}** · {r['image']}{src} · {stats()}")

    def act(i, text, status):
        if i >= len(queue):
            return (i, *show(i))
        r = queue[i]
        label = None
        if status != "rejected":
            label = parse_output(text)
            if label is None:
                img, t, h, _ = show(i)
                return i, img, text, h, "⚠️ JSON 형식/값이 올바르지 않습니다. 고친 뒤 다시 저장하세요."
            if status == "approved" and r.get("label") != label:
                status = "edited"
        append_jsonl(args.out, {"image": r["image"], "video": r.get("video"), "status": status,
                                "label": label, "teacher_label": r.get("label")})
        return (i + 1, *show(i + 1))

    with gr.Blocks(title="SafeStep 라벨 검수") as demo:
        idx = gr.State(0)
        gr.Markdown("박스 색: 🟥 가까움 · 🟨 보통 · 🟦 멀리 — 위험도·위험요소·안내문이 사진과 맞는지 보고, "
                    "틀리면 JSON을 고친 뒤 **승인/저장**. 라벨로 쓸 수 없는 장면이면 **폐기**.")
        with gr.Row():
            img = gr.Image(label="장면 (정답 박스 표시)", type="filepath", height=480)
            with gr.Column():
                box = gr.Textbox(label="라벨(JSON) — 필요하면 직접 수정", lines=16)
                hint = gr.Textbox(label="정답 박스 요약", interactive=False)
        info = gr.Markdown()
        with gr.Row():
            b_ok = gr.Button("✅ 승인/저장", variant="primary")
            b_rej = gr.Button("🗑️ 폐기 (부적합 장면)")
        outs = [idx, img, box, hint, info]
        b_ok.click(lambda i, t: act(i, t, "approved"), [idx, box], outs)
        b_rej.click(lambda i, t: act(i, t, "rejected"), [idx, box], outs)
        demo.load(lambda: (0, *show(0)), None, outs)

    # Gradio는 작업 폴더 밖 파일을 기본 차단 → 프레임·미리보기 폴더를 허용 목록에 추가
    demo.launch(share=args.share, allowed_paths=[str(frames.resolve()), str(RENDER_DIR)])


if __name__ == "__main__":
    main()
