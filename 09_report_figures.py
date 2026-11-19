"""[9단계] 발표용 그림·분석 생성.

입력 : outputs/eval/{base3b,lora_r16,teacher7b}_{metrics.json,pred.jsonl}, data/splits/stats.json
출력 : outputs/figures/
  01_model_comparison.png   주요 지표 막대그래프 (Base / LoRA / Teacher)
  02_risk_confusion.png     위험도 혼동행렬 (Base vs LoRA)
  03_hazard_per_class.png   위험요소 종류별 F1 (Base vs LoRA)
  04_data_distribution.png  split별 위험도 분포 + 학습셋 위험요소 빈도
  05_cases_success.png / 06_cases_failure.png   성공·실패 사례 (LoRA 예측 vs 정답)
  analysis.md               위 수치 요약 (발표 원고용)

사용 예:
    python 09_report_figures.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import RISK_LEVELS, read_jsonl
from metrics import rouge_l_char

plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EVAL = Path("outputs/eval")
OUT = Path("outputs/figures")
FRAMES = Path("data/frames")
MODELS = [("base3b", "Base 3B (zero-shot)"), ("lora_r16", "3B + LoRA (ours)"), ("teacher7b", "Teacher 7B (zero-shot)")]
COLORS = ["#9aa5b1", "#2f6fdf", "#e0913a"]


def load_metrics():
    return {n: json.loads((EVAL / f"{n}_metrics.json").read_text(encoding="utf-8")) for n, _ in MODELS}


def load_preds(name):
    return read_jsonl(EVAL / f"{name}_pred.jsonl")


def fig_comparison(ms):
    keys = [("json_valid", "JSON 형식"), ("risk_acc", "위험도 정확도"), ("risk_macro_f1", "위험도 macro-F1"),
            ("hazard_f1", "위험요소 F1"), ("position_acc", "위치 정확도"), ("guidance_rougeL", "안내문 ROUGE-L")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.27
    for i, (n, label) in enumerate(MODELS):
        vals = [ms[n].get(k) or 0 for k, _ in keys]
        xs = [j + (i - 1) * w for j in range(len(keys))]
        bars = ax.bar(xs, vals, w, label=label, color=COLORS[i])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(keys)), [k[1] for k in keys])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("점수 (높을수록 좋음)")
    ax.set_title("테스트셋 100장 (검수 라벨) — 모델 비교")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "01_model_comparison.png", dpi=170)
    plt.close(fig)


def confusion(recs):
    c = {a: {b: 0 for b in RISK_LEVELS} for a in RISK_LEVELS}
    for r in recs:
        if r["pred"]:
            c[r["ref"]["risk"]][r["pred"]["risk"]] += 1
    return c


def fig_confusion(preds):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, name, title in zip(axes, ["base3b", "lora_r16"], ["Base 3B", "3B + LoRA"]):
        c = confusion(preds[name])
        mat = [[c[a][b] for b in RISK_LEVELS] for a in RISK_LEVELS]
        ax.imshow(mat, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, mat[i][j], ha="center", va="center",
                        color="white" if mat[i][j] > 25 else "black", fontsize=11)
        ax.set_xticks(range(3), RISK_LEVELS)
        ax.set_yticks(range(3), RISK_LEVELS)
        ax.set_xlabel("예측")
        ax.set_ylabel("정답")
        ax.set_title(title)
    fig.suptitle("위험도 혼동행렬 (테스트 100장)")
    fig.tight_layout()
    fig.savefig(OUT / "02_risk_confusion.png", dpi=170)
    plt.close(fig)


def per_class(recs):
    tp, fp, fn = Counter(), Counter(), Counter()
    for r in recs:
        rt = {h["type"] for h in r["ref"]["hazards"]}
        pt = {h["type"] for h in r["pred"]["hazards"]} if r["pred"] else set()
        for t in pt & rt:
            tp[t] += 1
        for t in pt - rt:
            fp[t] += 1
        for t in rt - pt:
            fn[t] += 1
    out = {}
    for t in set(tp) | set(fn):
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0
        rc = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0
        out[t] = {"f1": 2 * p * rc / (p + rc) if p + rc else 0, "support": tp[t] + fn[t], "p": p, "r": rc}
    return out


def fig_per_class(preds):
    base, lora = per_class(preds["base3b"]), per_class(preds["lora_r16"])
    types = sorted(lora, key=lambda t: -lora[t]["support"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    xs = range(len(types))
    ax.bar([x - 0.2 for x in xs], [base.get(t, {"f1": 0})["f1"] for t in types], 0.4, label="Base 3B", color=COLORS[0])
    ax.bar([x + 0.2 for x in xs], [lora[t]["f1"] for t in types], 0.4, label="3B + LoRA", color=COLORS[1])
    ax.set_xticks(list(xs), [f"{t}\n(n={lora[t]['support']})" for t in types], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("위험요소 종류별 F1 (테스트 100장)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "03_hazard_per_class.png", dpi=170)
    plt.close(fig)
    return base, lora


def fig_data(stats):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    splits = ["train", "val", "test"]
    bottom = [0] * 3
    rc = {"높음": "#d9534f", "보통": "#f0ad4e", "낮음": "#5cb85c"}
    for lv in RISK_LEVELS:
        vals = [stats[s]["risk"].get(lv, 0) for s in splits]
        ax.bar(splits, vals, bottom=bottom, label=lv, color=rc.get(lv))
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, s in enumerate(splits):
        ax.text(i, bottom[i] + 15, f"{stats[s]['n']}장", ha="center", fontsize=9)
    ax.set_title("split별 위험도 분포 (시퀀스 단위 분할)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ht = stats["train"]["hazard_types"]
    ks = sorted(ht, key=lambda k: -ht[k])
    ax.barh(ks[::-1], [ht[k] for k in ks[::-1]], color=COLORS[1])
    ax.set_title("학습셋 위험요소 등장 횟수")
    fig.tight_layout()
    fig.savefig(OUT / "04_data_distribution.png", dpi=170)
    plt.close(fig)


def score(r):
    if not r["pred"]:
        return 0.0
    rt = {h["type"] for h in r["ref"]["hazards"]}
    pt = {h["type"] for h in r["pred"]["hazards"]}
    f1 = 1.0 if not rt and not pt else (2 * len(rt & pt) / (len(rt) + len(pt)))
    return (r["pred"]["risk"] == r["ref"]["risk"]) + f1 + rouge_l_char(r["pred"]["guidance"], r["ref"]["guidance"])


def fmt(lbl):
    if not lbl:
        return "(JSON 파싱 실패)"
    hz = ", ".join(f"{h['type']}/{h['position']}/{h['distance']}" for h in lbl["hazards"]) or "없음"
    g = lbl["guidance"]
    if len(g) > 26:
        g = g[:26] + "\n" + g[26:]
    return f"위험도: {lbl['risk']}\n요소: {hz}\n안내: {g}"


def fig_cases(recs, base_by_img, path, title):
    n = len(recs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.1 * n), gridspec_kw={"width_ratios": [1, 1.5]})
    for i, r in enumerate(recs):
        im = Image.open(FRAMES / r["image"]).convert("RGB")
        axes[i][0].imshow(im)
        axes[i][0].set_title(r["image"], fontsize=8)
        axes[i][0].axis("off")
        b = base_by_img.get(r["image"])
        txt = (f"[정답]\n{fmt(r['ref'])}\n\n[3B+LoRA]\n{fmt(r['pred'])}"
               + (f"\n\n[Base 3B]\n{fmt(b['pred'])}" if b else ""))
        axes[i][1].text(0, 1, txt, va="top", fontsize=7.5, family=plt.rcParams["font.family"])
        axes[i][1].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ms = load_metrics()
    preds = {n: load_preds(n) for n, _ in MODELS}
    stats = json.loads(Path("data/splits/stats.json").read_text(encoding="utf-8"))

    fig_comparison(ms)
    fig_confusion(preds)
    base_pc, lora_pc = fig_per_class(preds)
    fig_data(stats)

    lora = preds["lora_r16"]
    base_by_img = {r["image"]: r for r in preds["base3b"]}
    ranked = sorted(lora, key=score)
    fig_cases(ranked[-3:][::-1], base_by_img, OUT / "05_cases_success.png", "성공 사례 (3B + LoRA)")
    fig_cases(ranked[:3], base_by_img, OUT / "06_cases_failure.png", "실패 사례 (3B + LoRA)")

    # 횡단보도 인식 (정답에 횡단보도/가운데가 있는지)
    def cw(l):
        return bool(l) and any(h["type"] == "횡단보도" and h["position"] == "가운데" for h in l["hazards"])

    lines = ["# 평가 분석 요약 (자동 생성)", "", "## 모델 비교", "",
             (EVAL / "comparison.md").read_text(encoding="utf-8"), "## 위험도 혼동행렬 (행=정답, 열=예측)", ""]
    for name in ["base3b", "lora_r16", "teacher7b"]:
        c = confusion(preds[name])
        lines.append(f"- {name}: " + " / ".join(f"{a}→" + ",".join(f"{b}{c[a][b]}" for b in RISK_LEVELS) for a in RISK_LEVELS))
    lines += ["", "## 횡단보도(정면) 인식", ""]
    for name in ["base3b", "lora_r16", "teacher7b"]:
        tp = sum(cw(r["pred"]) and cw(r["ref"]) for r in preds[name])
        fp = sum(cw(r["pred"]) and not cw(r["ref"]) for r in preds[name])
        fn = sum((not cw(r["pred"])) and cw(r["ref"]) for r in preds[name])
        lines.append(f"- {name}: TP {tp} FP {fp} FN {fn}")
    lines += ["", "## 위험요소 종류별 F1 (Base → LoRA, support)", ""]
    for t in sorted(lora_pc, key=lambda t: -lora_pc[t]["support"]):
        lines.append(f"- {t}: {base_pc.get(t, {'f1': 0})['f1']:.2f} → {lora_pc[t]['f1']:.2f} (n={lora_pc[t]['support']})")
    lines += ["", "## 실패 사례 (점수 하위 3장)", ""]
    for r in ranked[:3]:
        lines.append(f"- {r['image']}: 정답 {json.dumps(r['ref'], ensure_ascii=False)}")
        lines.append(f"  - 예측 {json.dumps(r['pred'], ensure_ascii=False)}")
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n그림 저장 → {OUT}")


if __name__ == "__main__":
    main()
