"""[8단계] 데모 앱 (Gradio) — 발표 영상의 결과 데모용.

- 이미지 탭: 사진 1장 → 위험도 / 위험요소 / 음성 안내 (+ base 모델 결과와 나란히 비교)
- 동영상 탭: 보행 영상 → N초마다 프레임을 뽑아 안내 문장 타임라인 생성
- (선택) 음성 안내: 맥은 내장 `say` 명령(오프라인), 그 외는 gTTS

맥에서는 MLX로 base / LoRA 모델을 둘 다 올린다 (3B 4bit 두 개 ≈ 5GB).

사용 예:
    python 08_demo_app.py --adapter outputs/mlx-lora-r16
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import gradio as gr
from PIL import Image

from backend import Runner, default_backend
from common import parse_output

RISK_EMOJI = {"높음": "🔴 높음", "보통": "🟡 보통", "낮음": "🟢 낮음"}


def fmt(raw: str) -> str:
    p = parse_output(raw)
    if p is None:
        return f"⚠️ 형식 오류\n\n```\n{raw}\n```"
    hz = "\n".join(f"- {h['type']} · {h['position']} · {h['distance']}" for h in p["hazards"]) or "- 없음"
    return f"**위험도**: {RISK_EMOJI[p['risk']]}\n\n**위험 요소**\n{hz}\n\n**🔊 안내**: {p['guidance']}"


def tts(text: str):
    try:
        if platform.system() == "Darwin":
            f = tempfile.NamedTemporaryFile(suffix=".aiff", delete=False).name
            subprocess.run(["say", "-v", "Yuna", "-o", f, text], check=True)
            return f
        from gtts import gTTS

        f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        gTTS(text, lang="ko").save(f)
        return f
    except Exception as e:
        print("TTS 실패:", e)
        return None


def example_images(jsonl, frames, n=8):
    """위험도별로 골고루 테스트 이미지 n장 (데모 녹화용)."""
    from common import read_jsonl
    import json
    from pathlib import Path
    if not Path(jsonl).exists():
        return []
    by = {}
    for r in read_jsonl(jsonl):
        by.setdefault(json.loads(r["target"])["risk"], []).append(str(Path(frames) / r["image"]))
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by.values()):
        for v in by.values():
            if i < len(v) and len(out) < n:
                out.append(v[i * 7 % len(v)])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=default_backend(), choices=["mlx", "torch"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--no-base", action="store_true", help="base 비교용 모델을 올리지 않음(메모리 절약)")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--examples", default="data/splits/test.jsonl", help="예시 이미지로 보여줄 jsonl (테스트셋)")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    # MLX는 연산 스트림이 스레드별이라, Gradio 작업 스레드에서 바로 부르면
    # "There is no Stream(cpu, 0) in current thread" 오류가 난다.
    # → 모델 로딩과 모든 추론을 전용 스레드 1개에서 실행.
    worker = ThreadPoolExecutor(max_workers=1)
    lora = worker.submit(Runner, args.backend, model_id=args.model, adapter=args.adapter).result()
    base = None if args.no_base else worker.submit(Runner, args.backend, model_id=args.model).result()

    def run(runner, img):
        return worker.submit(runner.generate, img).result()

    def on_image(img, compare, speak):
        if img is None:
            return "", "", None
        raw = run(lora, img)
        base_md = fmt(run(base, img)) if (compare and base) else ""
        p = parse_output(raw)
        audio = tts(p["guidance"]) if (speak and p) else None
        return fmt(raw), base_md, audio

    def on_video(path, every):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(fps * every))
        gallery, lines, i = [], [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                p = parse_output(run(lora, img))
                t = i / fps
                msg = f"[{t:5.1f}s] {RISK_EMOJI[p['risk']]} {p['guidance']}" if p else f"[{t:5.1f}s] (형식 오류)"
                lines.append(msg)
                gallery.append((img, msg))
                yield gallery, "\n".join(lines)  # 프레임 하나 끝날 때마다 화면에 바로 표시
            i += 1
        cap.release()

    with gr.Blocks(title="SafeStep: 보행 안내 VLM") as demo:
        gr.Markdown("# 🦯 SafeStep — 보행 장면 한국어 안내 VLM\n"
                    "Qwen2.5-VL-3B (4bit) + LoRA · Teacher 7B 지식 증류 · Apple Silicon(MLX)")
        with gr.Tab("이미지"):
            with gr.Row():
                inp = gr.Image(type="pil", label="보행자 시점 사진")
                with gr.Column():
                    gr.Markdown("### LoRA (ours)")
                    out_lora = gr.Markdown()
                    gr.Markdown("### Base (학습 전)")
                    out_base = gr.Markdown()
                    audio = gr.Audio(label="음성 안내", autoplay=True)
            with gr.Row():
                compare = gr.Checkbox(base is not None, label="Base 모델과 비교", interactive=base is not None)
                speak = gr.Checkbox(False, label="음성 안내")
                btn = gr.Button("분석", variant="primary")
            btn.click(on_image, [inp, compare, speak], [out_lora, out_base, audio])
            ex = example_images(args.examples, args.frames)
            if ex:
                gr.Examples(ex, inputs=inp, label="테스트셋 예시 (학습에 쓰지 않은 시퀀스)")
        with gr.Tab("동영상"):
            vid = gr.Video(label="보행 영상")
            every = gr.Slider(0.5, 5, value=2, step=0.5, label="분석 간격(초)")
            vbtn = gr.Button("분석", variant="primary")
            gal = gr.Gallery(columns=4, height=420)
            log = gr.Textbox(label="안내 타임라인", lines=10)
            vbtn.click(on_video, [vid, every], [gal, log])

    demo.launch(share=args.share, server_port=args.port, allowed_paths=[args.frames])


if __name__ == "__main__":
    main()
