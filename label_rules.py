"""하이브리드 라벨링의 '규칙' 부분: 정답 박스 + 깊이 → 위험도·위험요소.

파일럿에서 Teacher(7B 4bit)는 위험도 기준을 줘도 20장 모두 '보통'으로 답했다.
그래서 구조화 필드(risk, hazards)는 사람이 단 정답 박스(AI Hub)와 깊이 추정으로 규칙 생성하고,
Teacher는 사진을 보고 자연어 안내문(guidance)만 쓰게 한다.
"""
from __future__ import annotations

MOVING = {"자동차", "오토바이", "자전거", "전동킥보드", "사람", "동물"}
NOT_OBSTACLE = {"신호등"}  # 머리 위에 있어 보행 장애물이 아님
DIST_RANK = {"가까움": 0, "보통": 1, "멀리": 2}
POS_RANK = {"가운데": 0, "왼쪽": 1, "오른쪽": 1}


def select_hazards(objects: list[dict], k: int = 3) -> list[dict]:
    """보행에 영향을 주는 것만 골라 심각한 순서로 최대 k개."""
    keep = []
    for o in objects:
        t, pos, dist = o["type"], o["position"], o["distance"]
        if t in NOT_OBSTACLE:
            continue
        if dist == "멀리" and not (pos == "가운데" and t in MOVING):
            continue
        if t == "기타" and dist != "가까움":
            continue
        keep.append(o)
    keep.sort(key=lambda o: (DIST_RANK[o["distance"]], POS_RANK[o["position"]], -o.get("depth", 0)))
    out, seen = [], set()
    for o in keep:  # 같은 종류·같은 방향은 가장 가까운 것 하나만
        key = (o["type"], o["position"])
        if key in seen:
            continue
        seen.add(key)
        # name: 안내 문장에 쓸 구체적 이름 (type이 '기타'여도 '전력제어함'처럼 말하기 위함, 라벨에는 저장 안 됨)
        out.append({"type": o["type"], "position": o["position"], "distance": o["distance"],
                    "name": o["type"] if o["type"] != "기타" else o.get("name", "장애물")})
        if len(out) == k:
            break
    return out


def risk_level(hazards: list[dict]) -> str:
    def any_(f):
        return any(f(h) for h in hazards)

    if any_(lambda h: h["distance"] == "가까움" and (h["position"] == "가운데" or h["type"] in MOVING)):
        return "높음"
    if any_(lambda h: h["distance"] == "가까움" or (h["distance"] == "보통" and
                                                   (h["position"] == "가운데" or h["type"] in MOVING))):
        return "보통"
    return "낮음"


DIR_WORD = {"가운데": "정면", "왼쪽": "왼쪽", "오른쪽": "오른쪽"}
DIST_WORD = {"가까움": "가까이", "보통": "조금 앞", "멀리": "멀리"}
FAST = {"자동차", "오토바이", "자전거", "전동킥보드"}
ACTION_KEY = {  # 행동 문장 → Teacher 출력에 반드시 남아 있어야 할 핵심어
    "멈추세요.": "멈추", "멈추고 지나갈 때까지 기다리세요.": "멈추",
    "왼쪽으로 비켜 가세요.": "왼쪽으로", "오른쪽으로 비켜 가세요.": "오른쪽으로",
    "주의하며 천천히 직진하세요.": "직진", "주의하며 직진하세요.": "직진", "그대로 직진하세요.": "직진",
}


def josa(word: str, with_batchim: str, without: str) -> str:
    ch = word[-1]
    if "가" <= ch <= "힣":
        return with_batchim if (ord(ch) - 0xAC00) % 28 else without
    return without


def _blocked(objects: list[dict], side: str) -> bool:
    return any(o["position"] == side and o["distance"] in ("가까움", "보통") and o["type"] not in NOT_OBSTACLE
               for o in objects)


def decide_action(risk: str, hazards: list[dict], objects: list[dict]) -> str:
    """사람이 단 박스 기준으로 행동을 정한다 (VLM이 위험한 방향으로 안내하는 오류 방지)."""
    if not hazards or risk == "낮음":
        return "그대로 직진하세요."
    top = hazards[0]
    if top["distance"] == "가까움" and top["type"] in FAST:
        return "멈추고 지나갈 때까지 기다리세요." if top["position"] == "가운데" else "주의하며 천천히 직진하세요."
    if top["distance"] == "가까움" and top["position"] == "가운데":
        left, right = _blocked(objects, "왼쪽"), _blocked(objects, "오른쪽")
        if left and right:
            return "멈추세요."
        return "오른쪽으로 비켜 가세요." if left else "왼쪽으로 비켜 가세요."
    return "주의하며 직진하세요."


def template_guidance(hazards: list[dict], action: str) -> str:
    if not hazards:
        return "앞이 비어 있습니다. " + action
    h = hazards[0]
    t = h.get("name") or h["type"]
    return f"{DIR_WORD[h['position']]} {DIST_WORD[h['distance']]}에 {t}{josa(t, '이', '가')} 있습니다. {action}"


def apply_crosswalk(label: dict) -> dict:
    """횡단보도 VQA가 '예'인 장면: 횡단보도를 맨 앞 위험요소로, 위험도 높음, 멈춰서 신호 확인."""
    hz = [h for h in label["hazards"] if h["type"] != "횡단보도"]
    hazards = [{"type": "횡단보도", "position": "가운데", "distance": "가까움"}] + hz[:2]
    near = [h for h in hz if h["distance"] == "가까움"]
    extra = ""
    if near:
        h = near[0]
        extra = f"{DIR_WORD[h['position']]} 가까이에 {h['type']}{josa(h['type'], '이', '가')} 있습니다. "
    guidance = "정면에 횡단보도가 있습니다. " + extra + "멈춰서 신호와 차량을 확인한 뒤 건너세요."
    return {"risk": "높음", "hazards": hazards, "guidance": guidance}


def guidance_prompt(template: str) -> str:
    return (
        "이 사진은 보행자 시점에서 촬영한 장면입니다. 아래는 시각장애인에게 들려줄 음성 안내 초안입니다.\n"
        f"초안: {template}\n\n"
        "사진을 보고 이 초안을 자연스러운 한국어 한두 문장으로 다듬으세요.\n"
        "- 초안의 방향(정면/왼쪽/오른쪽), 물체, 행동은 절대 바꾸지 마세요.\n"
        "- 횡단보도·자전거도로·차도·공사 구역처럼 길 상태가 사진에 보이면 짧게 덧붙여도 됩니다.\n"
        "- 따옴표나 설명 없이 안내 문장만 쓰세요."
    )


def clean_guidance(text: str, hazards: list[dict], action: str) -> str | None:
    """Teacher 출력 검증: 형식, 물체 이름, 행동 핵심어가 남아 있어야 통과."""
    s = text.strip().strip('"“”\'').replace("\n", " ").strip()
    if s.startswith("{") or len(s) < 6 or not any("가" <= ch <= "힣" for ch in s):
        return None
    if ACTION_KEY[action] not in s:
        return None
    if hazards:
        h = hazards[0]
        name = (h.get("name") or h["type"]).replace(" ", "")
        if name not in s.replace(" ", "") or DIR_WORD[h["position"]] not in s:
            return None
        wrong = {"왼쪽으로": "오른쪽으로", "오른쪽으로": "왼쪽으로"}.get(ACTION_KEY[action])
        if wrong and wrong in s:  # 반대 방향 안내가 섞이면 탈락
            return None
    return s[:200]
