# SafeStep — 보행 장면 한국어 안내 VLM

시각장애인 보행 보조를 위해, 보행자 시점 이미지/영상을 보고
**위험도 · 위험 요소(종류/위치/거리) · 한국어 음성 안내 문장**을 JSON으로 생성하는 Vision-Language Model.

- 라벨: AI Hub 정답 박스 + 깊이 규칙(위험도·위험요소·행동) → Teacher [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (4bit)가 안내문 다듬기 + 횡단보도 VQA → 테스트셋 100장 검수
- Student: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) (4bit) + **LoRA** 파인튜닝 (이미지만 입력)
- 실행 환경: **Apple Silicon 맥 + [MLX](https://github.com/ml-explore/mlx) / [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)**
  (4bit 변환본: [mlx-community/Qwen2.5-VL-3B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-VL-3B-Instruct-4bit), [7B-4bit](https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-4bit))
- 보조 모델: [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) (정답 박스가 없는 데이터용으로 [YOLO11](https://github.com/ultralytics/ultralytics) 경로도 있음)
- 데이터: [AI Hub 인도보행 영상](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=189) — 한국지능정보사회진흥원(NIA) AI 허브 데이터 활용
- 관련 연구: WalkVLM (arXiv 2412.20903)

## 링크
- 코드: https://github.com/hwa0907/safestep-vlm
- 라벨 데이터셋: 이 저장소의 [`data/splits/`](data/splits) (train 1,132 / val 160 / test 100, 검수 라벨 `data/reviewed.jsonl`, 횡단보도 VQA `data/crosswalk.jsonl`) — 원본 이미지는 미포함, `image` 파일명과 AI Hub 원본 경로로 매칭
- LoRA 어댑터: [Releases › v1.0](https://github.com/hwa0907/safestep-vlm/releases/tag/v1.0) 의 `safestep-lora-r16.zip` → `outputs/mlx-lora-r16/`에 풀고 `08_demo_app.py --adapter outputs/mlx-lora-r16`
- 원본 데이터: [AI Hub 인도보행 영상](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=189) (직접 신청 후 다운로드)

### 왜 맥에서 학습하나
AI Hub는 해외 IP 다운로드를 차단하고, 약관상 데이터의 국외 반출에 별도 합의가 필요하다.
Colab 서버는 해외에 있으므로 다운로드·라벨링·학습·평가를 **모두 국내 로컬 맥에서** 수행한다.
CUDA 전용인 bitsandbytes 대신 MLX의 4bit 양자화 + LoRA(QLoRA와 같은 구조)를 쓴다.
(CUDA GPU가 있는 환경을 위한 PyTorch 경로 `06_train_qlora.py`, `--backend torch` 도 남겨 둠)

## 파이프라인

```
AI Hub 인도보행(바운딩박스) ── 01b 이미지 가져오기 · 시퀀스별 샘플링 (1,452장)
   └─ 정답 박스 + 깊이     ── 02 CVAT 박스(29종) + Depth Anything V2 → 종류·위치·거리
        └─ 규칙 라벨        ── label_rules: 위험도 · 위험요소(최대 3) · 행동(비켜/멈춤) 결정
             └─ Teacher 7B  ── 03 안내문만 자연스럽게 다듬기 (물체·방향·행동이 바뀌면 규칙 문장으로 대체)
                  └─ 3b VQA ── 발 앞 crop "흰 줄무늬?" → 전체 사진 재확인 → 횡단보도 반영
                       └─ 검수 ── 04 테스트 100장 검수 (승인 64 / 수정 36)
                            └─ 분할 ── 05 시퀀스 단위 train 1,132 / val 160 / test 100
                                 └─ Student 3B LoRA ── 06_train_mlx (bf16 변환 모델, 2 epoch)
                                      └─ 평가 / 데모 / 그림 ── 07 비교표, 08 Gradio 데모, 09 발표 그림
```

## 결과 (테스트 100장, 검수 라벨)

| 모델 | JSON 유효 | 위험도 정확도 | 위험도 macro-F1 | 위험요소 F1 | 위치 정확도 | 안내문 ROUGE-L | 지연(초/장) | 최대 메모리 |
|---|---|---|---|---|---|---|---|---|
| Base 3B (zero-shot) | 0.97 | 0.43 | 0.20 | 0.23 | 0.32 | 0.37 | 3.6 | 3.7GB |
| **3B + LoRA (ours)** | **1.00** | **0.65** | **0.55** | **0.73** | **0.78** | **0.73** | 4.2 | 4.2GB |
| Teacher 7B (zero-shot) | 1.00 | 0.44 | 0.25 | 0.47 | 0.70 | 0.51 | 6.2 | 6.2GB |

- 학습: LoRA rank 16 / α 32, 학습 파라미터 29.9M (0.80%), 2 epoch 2,264 iteration, 약 3시간 30분 (중간 검증 포함, M5 16GB), 최대 메모리 5.9GB, val loss 0.124 → 0.094
- 그림: `outputs/figures/` (`09_report_figures.py`), 상세 분석: `outputs/figures/analysis.md`

## 맥에서 실행

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.13
pip install -r requirements-mac.lock.txt   # 실제 동작 확인한 전체 버전 고정 (요약판: requirements-mac.txt)
export MLX_MPI_LIBNAME=libmpi_disabled.dylib   # Anaconda 파이썬 기반일 때 MLX 학습 abort 방지

# 0) AI Hub 다운로드 (국내 PC에서만 가능) — 키는 화면에 안 보이게 입력
mkdir -p ~/aihub && cd ~/aihub && curl -s -o aihubshell https://api.aihub.or.kr/api/aihubshell.do && chmod +x aihubshell
read -s "AIHUB_KEY?API key: " && echo
./aihubshell -mode d -datasetkey 189 -filekey 49933 -aihubapikey "$AIHUB_KEY"
cd <이 프로젝트 폴더>

python 01b_import_images.py --src ~/aihub --out data/frames --stride 10 --max-per-group 40 --max-total 1500
python 02_aux_hints.py --gt-src ~/aihub/extracted     # 정답 박스 + 깊이
python 03_teacher_label.py --limit 20      # 품질·속도 먼저 확인 후
python 03_teacher_label.py --priority data/splits/split_assign.json   # 전체 (test → val → train 순)
python 03b_crosswalk_vqa.py                # 횡단보도 VQA
python 04_review_app.py --only data/splits/test.jsonl --shuffle   # 테스트셋 검수
python 05_build_splits.py --reviewed data/reviewed.jsonl --test-reviewed-only --allow-unreviewed --crosswalk
python tools_convert_bf16.py mlx-community/Qwen2.5-VL-3B-Instruct-4bit models/qwen25vl-3b-4bit-bf16   # 4bit 모델의 float16 → bfloat16 (NaN 방지, 1회)
python 06_train_mlx.py --out outputs/mlx-lora-r16
python 07_evaluate.py run --name base3b
python 07_evaluate.py run --name lora_r16 --adapter outputs/mlx-lora-r16
python 07_evaluate.py run --name teacher7b --role teacher
python 07_evaluate.py table --names base3b lora_r16 teacher7b
python 08_demo_app.py --adapter outputs/mlx-lora-r16
python 09_report_figures.py
```

## 출력 스키마

```json
{"risk": "높음", "hazards": [{"type": "전동킥보드", "position": "왼쪽", "distance": "가까움"}],
 "guidance": "왼쪽 가까이에 전동킥보드가 있습니다. 오른쪽으로 비켜 가세요."}
```

## 데이터 공개 범위 (AI Hub 약관)
- 원본 이미지는 제3자 제공 금지 → HF에는 **라벨(JSON) + 이미지 파일명 + AI Hub 원본 경로만** 업로드 (`05_build_splits.py --push-to-hub`).
- 학습된 LoRA 어댑터(2차 저작물)는 공개 가능.
- 발표·코드에 출처 표기: "한국지능정보사회진흥원(NIA) AI 허브 '인도보행 영상' 데이터 활용".

## 파일

| 파일 | 내용 |
|---|---|
| `common.py` | 프롬프트, 스키마, 파싱 (+ PyTorch 로딩) |
| `mlx_backend.py` / `backend.py` | MLX 추론, 백엔드 자동 선택(맥=mlx, CUDA=torch) |
| `metrics.py` | JSON 유효율, 위험도 정확도, 위험요소 F1, 위치 정확도, ROUGE-L/chrF |
| `label_rules.py` | 규칙 라벨: 위험요소 선택, 위험도, 행동, 안내문 검증, 횡단보도 반영 |
| `03b_crosswalk_vqa.py` | 횡단보도 VQA (crop → 전체 사진 2단계) |
| `tools_convert_bf16.py` | 4bit MLX 모델의 float16 텐서 → bfloat16 |
| `06_train_mlx.py` | 맥 LoRA 학습 (기본) · `06_train_qlora.py` CUDA용 |
| `09_report_figures.py` | 발표용 그림·분석 (비교 막대, 혼동행렬, 종류별 F1, 사례) |
| `docs/presentation_outline.md` | 5분 발표 구성안 |
| `docs/troubleshooting_log.md` | 문제·해결 기록 (평가 항목 10점) |
| `tests/test_cpu.py` | GPU 없이 도는 단위 테스트 |
