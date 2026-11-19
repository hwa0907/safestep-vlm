"""평가 지표 (torch 없이 동작 → 로컬에서도 테스트 가능).

- json_valid   : 출력이 스키마에 맞는 JSON인 비율 (형식 준수)
- risk_acc / risk_macro_f1 : 위험도(높음/보통/낮음) 분류 정확도
- hazard_p/r/f1 : 이미지별 위험요소 '종류' 집합에 대한 micro P/R/F1
- position_acc : 정답·예측 모두에 있는 종류에 대해 위치(왼/가운데/오른쪽)가 맞은 비율
- guidance_rougeL : 안내 문장 글자 단위 ROUGE-L F1 (한국어는 띄어쓰기 편차가 커서 글자 단위)
- guidance_chrf : 안내 문장 chrF (sacrebleu 설치 시)
"""
from __future__ import annotations

from common import RISK_LEVELS


def lcs(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l_char(pred: str, ref: str) -> float:
    pred, ref = pred.replace(" ", ""), ref.replace(" ", "")
    l = lcs(pred, ref)
    if l == 0:
        return 0.0
    p, r = l / len(pred), l / len(ref)
    return 2 * p * r / (p + r)


def compute(preds: list[dict | None], refs: list[dict]) -> dict:
    n = len(refs)
    valid = [p is not None for p in preds]
    tp = fp = fn = 0
    pos_hit = pos_tot = 0
    risk_hit = 0
    conf = {a: {b: 0 for b in RISK_LEVELS} for a in RISK_LEVELS}
    rl, g_pred, g_ref = [], [], []
    for p, r in zip(preds, refs):
        rt = {h["type"]: h for h in r["hazards"]}
        if p is None:
            fn += len(rt)
            rl.append(0.0)
            g_pred.append("")
            g_ref.append(r["guidance"])
            continue
        pt = {h["type"]: h for h in p["hazards"]}
        tp += len(pt.keys() & rt.keys())
        fp += len(pt.keys() - rt.keys())
        fn += len(rt.keys() - pt.keys())
        for t in pt.keys() & rt.keys():
            pos_tot += 1
            pos_hit += pt[t]["position"] == rt[t]["position"]
        risk_hit += p["risk"] == r["risk"]
        conf[r["risk"]][p["risk"]] += 1
        rl.append(rouge_l_char(p["guidance"], r["guidance"]))
        g_pred.append(p["guidance"])
        g_ref.append(r["guidance"])

    # 정답·예측 모두 위험요소가 하나도 없으면(빈 장면을 빈 장면이라고 맞힘) 완벽한 것으로 본다
    if tp + fp + fn == 0:
        prec = rec = 1.0
    else:
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
    f1s = []
    for c in RISK_LEVELS:
        ctp = conf[c][c]
        cfp = sum(conf[o][c] for o in RISK_LEVELS if o != c)
        cfn = sum(conf[c][o] for o in RISK_LEVELS if o != c)
        f1s.append(2 * ctp / (2 * ctp + cfp + cfn) if ctp else 0.0)

    out = {
        "n": n,
        "json_valid": sum(valid) / n,
        "risk_acc": risk_hit / n,
        "risk_macro_f1": sum(f1s) / len(f1s),
        "hazard_p": prec,
        "hazard_r": rec,
        "hazard_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "position_acc": pos_hit / pos_tot if pos_tot else None,  # 맞힌 위험요소가 없으면 계산 불가
        "guidance_rougeL": sum(rl) / n,
        "risk_confusion": conf,
    }
    try:
        import sacrebleu

        out["guidance_chrf"] = sacrebleu.corpus_chrf(g_pred, [g_ref]).score / 100
    except ImportError:
        pass
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
