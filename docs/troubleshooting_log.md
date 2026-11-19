# 문제·해결 기록 (평가 항목: 10점)

> 작업하면서 **사소한 오류까지 바로 한 줄씩** 적는다. 발표 때 5~6개를 골라 표로 보여준다.
> 아래 "예상 이슈"는 미리 대비해 둔 것으로, 실제로 겪으면 날짜와 실제 메시지를 채워 넣는다.

## 기록

| 날짜 | 단계 | 증상(오류 메시지) | 원인 | 해결 |
|---|---|---|---|---|
| 2026-11-09 | 로컬 테스트 | `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.5` / `_ARRAY_API not found` (import cv2) | Anaconda base 환경의 OpenCV 바이너리가 NumPy 1.x용, 설치된 NumPy는 2.x | base는 건드리지 않고 프로젝트 전용 가상환경 생성 후 최신 `opencv-python-headless` 설치 → 해결 |
| 2026-11-09 | 1b단계 | 한글 경로 이미지에서 `cv2.imread`가 None 반환 가능 | OpenCV imread의 비ASCII 경로 처리 문제(특히 Windows) | `np.fromfile` + `cv2.imdecode`로 읽기 |
| 2026-11-11 | Colab 설정 | `No such file or directory: 'requirements.txt'` | 노트북만 열려 있고 Drive에 프로젝트 폴더가 없음 / 작업 위치 불일치 | 첫 셀에서 zip 업로드 → Drive에 압축 해제 → 프로젝트 폴더로 `os.chdir` |
| 2026-11-11 | Colab 설정 | pip `dependency conflicts` (diffusers↔huggingface-hub, spacy/wandb↔click) | transformers 4.51.3 고정이 Colab 기본 패키지보다 낮은 huggingface-hub를 설치 | 충돌 패키지(diffusers, spacy, wandb)는 이 프로젝트에서 안 쓰므로 무시. 런타임 재시작 후 진행 |
| 2026-11-11 | Colab 설정 | "자동으로 저장할 수 없습니다. 이 파일은 원격으로 또는 다른 탭에서 업데이트되었습니다." | 열어 둔 노트북 파일을 `unzip -o`가 같은 경로에 덮어씀 | 페이지 새로고침으로 Drive의 최신 노트북 다시 열기 |
| 2026-11-11 | 0단계 | `이미지가 없습니다: /content/aihub` → 원인 출력 추가 후 `HTTP status 502 ... AI 허브는 해외에서의 데이터 다운로드를 제한하고 있습니다.` (aihubshell 종료 코드는 0이라 실패가 안 보였음) | Colab 서버가 해외(미국)에 있어 AI Hub가 차단 | 국내 PC(맥)에서 다운로드. 실패해도 0을 돌려주는 도구라 출력 문구로 성공 여부 확인 |
| 2026-11-11 | 환경 결정 | (약관 검토) AI Hub 약관 3항 "국외 반출 시 별도 합의 필요" | Colab에 데이터를 올리면 해외 서버로 반출됨 | 다운로드·라벨링·학습·평가를 전부 로컬 맥(M5 16GB)에서 수행. CUDA 전용 bitsandbytes 대신 Apple MLX(mlx-vlm)의 4bit + LoRA 사용 |
| 2026-11-11 | 6단계(맥) | 학습 시작 직후 `[mpi] MPI found but it does not appear to be Open MPI ... MPICH` 후 종료 코드 -6 (abort) | Anaconda 파이썬의 라이브러리 경로에 MPICH가 있어 MLX 분산 초기화가 그걸 로드 | 1차: 분산 백엔드를 ring으로 강제 → 4번째 iteration의 all_sum에서 다시 abort. 2차: `MLX_MPI_LIBNAME=libmpi_disabled.dylib`로 MPI 비활성 → 해결 |
| 2026-11-11 | 6단계(맥) | 학습은 됐는데 loss 그래프가 비어 있음 ("loss 기록 train 0개") | mlx-vlm 로그 형식이 `Training (iteration N):` 다음 줄에 `train_loss=...` | 헤더 줄에서 iteration을 기억하고 다음 줄의 `train_loss=`/`val_loss=`를 읽도록 파서 수정 |
| 2026-11-11 | 0단계(맥) | 다운로드가 끝났는데 `Bbox_10_new.zip`이 정확히 1,073,741,824바이트(1GiB, 원래 9.59GB) | aihubshell은 10GB를 tar 하나로 받은 뒤 1GiB 조각(part)들을 병합하는데, 다운로드가 중간에 끊겨 첫 조각만 남은 채 "병합"됨 | 불완전 zip 삭제 후 `caffeinate`로 절전 방지하고 재다운로드, 완료 후 용량·`unzip -t`로 검증 |
| 2026-11-11 | 7단계 | 빈 장면을 빈 장면이라고 맞혔는데 hazard_f1 = 0 | 정답·예측 모두 위험요소 0개 → TP=FP=FN=0 이라 0으로 계산됨 | 이 경우 P=R=F1=1.0, 위치 정확도는 계산 불가(None)로 처리 + 단위 테스트 추가 |
| 2026-11-11 | 4단계 | 검수 화면 모든 칸에 "오류", 서버 로그 `InvalidPathError: Cannot move ... to the gradio cache dir` | Gradio 6는 작업 폴더·임시 폴더 밖의 파일을 보안상 차단 | `launch(allowed_paths=[프레임 폴더])` |
| 2026-11-11 | 8단계(맥) | 데모에서 분석 누르면 `RuntimeError: There is no Stream(cpu, 0) in current thread` | MLX 연산 스트림은 스레드별인데 Gradio는 요청마다 다른 작업 스레드에서 함수 실행 | 모델 로딩과 추론을 전용 스레드 1개(ThreadPoolExecutor)에서만 실행 |
| 2026-11-11 | 0단계(맥) | 재다운로드도 정확히 9GiB(9×1GiB)에서 끝나고 `unzip -l`: "cannot find zipfile directory", 파일 끝이 0으로 채워짐 | 다운로드가 마지막 조각 전에 끊김. zip은 목차(central directory)가 파일 맨 끝에 있어 통째로 못 엶 | 세 번째 다운로드 대신 `bsdtar`로 앞에서부터 순차 추출 → 이미지 10,643장 + XML 38개 복구 (마지막 1장만 손상, 삭제). 필요한 1,452장에 충분 |
| 2026-11-11 | 2단계 | 가까운 차·가로수가 '멀리'로 분류됨 (가까움 8% / 보통 22% / 멀리 70%) | Depth Anything 상대깊이 임계값(170/90)이 너무 엄격 | 객체 7,834개 깊이 분포(p50=60, p85=130)와 육안 확인으로 140/70으로 보정 + 박스가 화면 하단에 닿으면 '가까움' 규칙 추가 → 14% / 28% / 58% |
| 2026-11-11 | 3단계 | 파일럿 20장: Teacher(7B 4bit)가 위험도를 20장 모두 '보통'으로 답함. 기준을 프롬프트에 명시해도 그대로 | 4bit 7B 모델이 다조건 규칙을 잘 따르지 못함 | **하이브리드 라벨링**: 위험도·위험요소는 정답 박스+깊이로 규칙 생성 → 낮음 30% / 보통 46% / 높음 24% |
| 2026-11-11 | 3단계 | 파일럿 2차: Teacher 안내문이 위험한 방향으로 안내 ("오른쪽 가까이에 자동차… 오른쪽으로 비켜 가세요"), 방향 착각, 자기모순 | VLM의 공간 추론·지시 따르기 한계 | 행동도 규칙으로 결정(좌우 빈 쪽 계산, 차량 근접 시 멈춤), Teacher는 문장 다듬기만. 출력에서 물체·방향·행동이 바뀌면 탈락시키고 규칙 문장 사용 → 파일럿 20장 중 16장 Teacher 문장 채택, 4장 대체 |
| 2026-11-11 | 3단계 | 안내문에 "기타가 있습니다" | 어휘에 없는 물체(전력제어함 등)를 '기타'로 묶음 | 라벨은 '기타'로 두고, 안내문에는 정답 박스의 세부 이름 사용 |
| 2026-11-11 | 4단계 | 테스트 100장 검수(Claude 검수, reviewer="claude"로 기록): 승인 64 / 수정 36. 수정 사유: 횡단보도·차도 누락 15, 문장 표현 7, 가장자리 물체 위치 판정 6, 방향·행동 오류 5, 원본 박스 오류·누락 3 | AI Hub 바운딩박스 29종에는 **노면 정보(횡단보도·차도)가 없음** → 규칙이 바로 앞 횡단보도를 '위험 없음'으로 판정. 화면 3등분 위치 규칙은 보행로 가장자리 물체를 '가운데'로 봄. 좌우 빈 공간 판단은 차도/상점 물건을 모름 | 테스트셋은 검수 라벨로 고정. 학습 라벨 개선안: Teacher에게 "바로 앞에 횡단보도가 있나?" 예/아니오만 묻는 VQA 패스 추가 |
| 2026-11-11 | 3b단계 | 횡단보도 VQA(전체 사진 질문): 재현율 1.00인데 정밀도 0.29 (160장 중 67장 '예') | 사진 어딘가에 횡단보도가 보이면(옆 차도, 먼 곳) '예'라고 답함 | **ROI crop**: 발 앞 영역(가로 20~80%, 세로 55~100%)만 잘라 "흰 줄무늬가 보이나?" 질문 → 정밀도 0.60, 정확도 0.92 |
| 2026-11-11 | 3b단계 | crop 방식의 남은 오탐 8건 | 줄무늬 보도블록·점자블록을 횡단보도 줄무늬로 착각 | crop '예'일 때만 전체 사진으로 재확인(AND) → **정밀도 0.75, 재현율 1.00, 정확도 0.96** (검수 100장 기준) |
| 2026-11-11 | 6단계(맥) | 학습 loss가 0.77→0.16으로 잘 내려가다 **150 iteration에서 갑자기 NaN**, 이후 전부 NaN (발견까지 25분 낭비) | mlx-community 4bit 변환본: config는 bfloat16인데 비양자화 가중치·scale 1,077개가 실제로는 **float16** → 최대 약 65,504를 넘는 활성값/기울기가 inf→NaN | `tools_convert_bf16.py`로 F16 텐서를 bfloat16으로 변환한 로컬 모델로 학습 + 학습 스크립트에 NaN 감지 시 즉시 중단 기능 추가 |
| 2026-11-12 | 6단계(맥) | (재학습 결과) bf16 모델로 2 epoch 완주, NaN 0회. val loss 0.124 → 0.106 → 0.097 → 0.094 | — | 과적합 없이 수렴. iteration당 약 4.8초, 최대 메모리 5.9GB |
| 2026-11-12 | 7단계 | Base 3B는 유효 출력 97장 전부 위험도 '높음', Teacher 7B는 99장이 '보통' → 정확도 0.43/0.44는 사실상 "다수 클래스 찍기" | zero-shot 모델이 위험도 기준을 이미지에 적용하지 못하고 한 값으로 붕괴 (파일럿에서 본 현상과 같음) | 정확도만 보면 차이가 작아 보여서 **macro-F1과 혼동행렬을 함께 보고** (Base 0.20 / Teacher 0.25 / LoRA 0.55) |
| 2026-11-12 | 7단계 | LoRA 오류 유형: 정답 '보통'을 '높음'으로 23건, '낮음'을 '보통'으로 9건 (과대평가 쪽) / '기타'·'바리케이드' F1 0 | 학습셋에서 '기타' 88건, '바리케이드' 8건으로 희소 + '기타'는 여러 물체를 묶은 클래스라 시각적으로 일관성이 없음 | 보행 안내에서는 과소평가보다 과대평가가 안전한 쪽이라 허용. 향후: 희소 클래스 오버샘플링, '기타' 세분화 |
| 2026-11-12 | 8단계 | 데모에서 예시 이미지를 누른 뒤 '분석' 버튼이 눌리지 않음 | 이미지가 들어오면서 레이아웃이 내려가 버튼 위치가 바뀜 | 데모 녹화 때 예시 클릭 후 버튼 위치 확인. 테스트셋 예시 8장(위험도별 골고루)을 데모에 추가해 녹화 준비 시간 단축 |

## 예상 이슈와 대응

| 증상 | 원인 | 해결 |
|---|---|---|
| `KeyError: 'qwen2_5_vl'` / 모델 클래스 import 실패 | transformers 버전이 낮음 | `transformers>=4.49` (requirements 고정 버전 사용) |
| T4에서 `bf16 is not supported` | T4(Turing)는 bfloat16 미지원 | `pick_dtype()`이 자동으로 fp16 선택 |
| fp16 학습 중 loss가 `nan` | fp16 overflow | lr 1e-4로 낮추기, L4/A100(bf16)으로 전환 |
| `CUDA out of memory` | 이미지 토큰 수가 해상도에 비례 (28×28 패치) | `--max-pixels` 축소(448²→336²), batch 1 + grad-accum 16 |
| NumPy 2.x 와 OpenCV 충돌 (`compiled using NumPy 1.x`) | 바이너리 호환성 (수업 10주차 실습과 동일) | `numpy==1.26.4`로 내리거나 최신 opencv로 올림 |
| Teacher 출력이 JSON이 아님 (설명문, ```json 블록) | 생성형 모델의 형식 이탈 | 펜스 제거 후 `{...}` 추출, 실패 시 sampling 재시도 |
| Teacher가 없는 물체를 만들어냄 (hallucination) | VLM의 과잉 생성 | YOLO+깊이 힌트 제공, 사람 검수로 수정 |
| 검증 성능이 비정상적으로 높음 | 프레임 무작위 분할 → 같은 장면 누수 | 영상 단위 분할(`05_build_splits.py`) |
| Colab 세션 끊김으로 라벨링 중단 | 런타임 제한 | 결과를 한 줄씩 append, 재실행 시 완료분 skip |
| `ValueError: ... unused columns` / 이미지가 collator에 안 들어옴 | Trainer가 모델 forward에 없는 컬럼 삭제 | `remove_unused_columns=False` |
| LoRA가 vision tower까지 붙어 메모리 증가 | target_modules 이름이 vision MLP와 겹침 | 정규식으로 `visual` 경로 제외 |
| 4bit 모델 학습 시 grad 없음 오류 | k-bit 모델의 입력 임베딩 grad 비활성 | `prepare_model_for_kbit_training` |
| 동영상 데모가 느림 | 프레임마다 생성 | 분석 간격(초) 슬라이더, max_new_tokens 제한 |
