# Fall Agent 낙상 감지기 — `synth_v1`

**2026-11-27 기준 채택 경로는 팀이 생성한 MuJoCo IMU로만 학습한 1D CNN이다.** CNN은 낙상 의심 사건을 대화 에이전트에 넘기는 앞단이지, 사용자 상태나 신고 여부를 결정하는 모델이 아니다. 동봉한 [원 체크포인트](weights/net_synth_v1.pt)로 SisFall 원시 기록을 다시 평가한 결과, 낙상 673건 중 390건을 감지했고 일상동작 1,522건 중 5건에 오경보를 냈다. **재현한 것은 오프라인 파일 단위 결과**이며 Jetson 현장 성능은 아니다.

## 무엇이 최종 경로인가

| 구분 | 데이터와 용도 |
|---|---|
| 채택한 `synth_v1` | 정지 출발 낙상 3종과 일상동작(ADL) 5종을 직접 생성. 학습·내부 검증에는 이 합성 기록만 사용. SisFall은 학습 입력이 아니다. |
| 비교 실험 | 원 실험의 SisFall SA01–15 학습 모델은 같은 시험군에서 Recall 99.1%, FPR 1.38%, AUC 0.999를 기록했지만 **실측 학습 비교군**이다. `capacity_*`, `sis_aug_v1`, `hard_adl_v2`, `paired_slots_v1`, `night_transition_v3`, `night_followup`의 학습·증강·후처리 결과도 채택 모델 성능이 아니다. HAPT 미사용. |
| 향후 계획 | Isaac 가상 IMU, 직접 수집한 MPU6050, Jetson ONNX/TensorRT 측정은 안 A의 후속 과제다. 이 가중치의 학습 데이터나 이 문서의 측정 결과에 포함되지 않는다. |

## 데이터가 CNN 입력이 되기까지

1. **생성** — [`synth_v1.py`의 `build`, `assist`, `run`](scripts/synth_v1.py)는 공식 humanoid 골반에 3축 가속도계·자이로를 붙인다. 5초를 2 ms 간격(500 Hz)으로 적분하고 처음 0.8초를 제외해 기록당 `(2100,6)`을 얻는다. 가속도는 `m/s² ÷ 9.81 → g`, 각속도는 `rad/s → °/s`. 질량·접촉·강성·힘과 방향을 바꾼다. 낙상은 `topple`, `trip_like`, `limp`; 일상은 `stand`, `stumble_recover`, `squat`, `bend`, `jump`다. 최종 몸통 높이가 낙상 `<0.45 m`, 일상 `>1.0 m`인 유한 궤적만 채택한다. `lie_slow` 분기는 생성 목록에 없다.
2. **추출** — 종류별 `100000 × 종류 인덱스 + 시도 번호`를 seed로 480번 시도해 앞에서부터 유효한 300개씩, 총 2,400기록(낙상 900, 일상 1,500)을 보존했다. NPZ: `X`(6축), `y`(1/0), `kind`, `imp`(500 Hz 낙상 가속도 크기 최댓값 인덱스, 일상 −1). [`synth_windows`](scripts/train_eval.py#L25)는 연속 5개 축별 평균으로 100 Hz로 낮춘다. 기록당 3초 창을 4개 뽑되 낙상은 피크가 창 시작 1.5–2.5초에 들어가야 한다. 조건을 못 맞춘 낙상 36기록은 창을 만들지 못했다. 일상은 시작점을 균등 추출한다. 가속도에 gain 0.95–1.05, 표본 잡음과 축별 bias(각 σ=0.02 g)를 더하고 ±16 g로 자른다. 자이로 잡음 σ=0.5 °/s. 합성 검증 창에도 같은 추출 절차를 적용했다.
3. **특징** — [`feats`](scripts/train_eval.py#L11)는 창마다 `(3,300)`을 만든다. 채널은 ① `‖a‖`(g), ② `‖ω‖ / 100`(°/s를 100으로 나눈 값), ③ 최근 최대 25표본 평균 가속도와 **창 첫 50표본 평균 가속도** 사이의 각도(rad)다. 세 번째 값은 절대 자세나 센서의 기울기 정답이 아니다. 원시 6축이 그대로 CNN에 들어가지는 않는다.
4. **분할** — [`train_synth.py`의 `split`](scripts/train_synth.py)는 NumPy seed 0으로 창 추출에 사용한 RNG를 이어 사용한다. 창을 무작위로 나누지 않고 유효 시뮬레이션 ID의 1/5(내림)을 검증으로 고른다. 같은 궤적의 네 창은 반드시 한쪽에만 있다.

| 분할 | 기록 수 | 낙상 창 | 일상 창 | 합계 창 |
|---|---:|---:|---:|---:|
| 합성 학습 | 1,892 | 2,852 | 4,716 | 7,568 |
| 합성 내부 검증 | 472 | 604 | 1,284 | 1,888 |

## 네트워크와 손실

[`Net.__init__` / `forward`](scripts/train_eval.py#L43) 기준 **학습 파라미터 32,001개**. 아래 크기는 배치 축을 생략한 `채널 × 시간`이다. Conv stride는 1이며 풀링 전까지 시간 길이를 유지한다.

| 순서 | 연산 | 출력 |
|---|---|---|
| 입력 | 100 Hz × 3초, 특징 3개 | 3 × 300 |
| 1 | Conv1d 3→32 (k=7, pad=3) → BN → ReLU → MaxPool(2) | 32 × 150 |
| 2 | Conv1d 32→64 (k=5, pad=2) → BN → ReLU → MaxPool(2) | 64 × 75 |
| 3 | Conv1d 64→64 (k=5, pad=2) → BN → ReLU | 64 × 75 |
| 헤드 | 시간축 평균과 최댓값을 이어 붙임 → Linear(128,1) | 창당 logit 1개 |

[`train`](scripts/train_eval.py#L52)은 `BCEWithLogitsLoss(pos_weight=4716/2852)`를 쓴다. 양성 항에만 약 1.654배 가중치를 주며 sigmoid를 손실 전에 중복 적용하지 않는다. Adam(lr=0.001), 배치 128, 15 epoch, epoch마다 무작위 순열, Torch seed 0이다. 마지막 epoch의 state_dict를 저장하며 검증 최고점 체크포인트를 고르지 않는다. 학습 당시 PyTorch 2.8.0 / CUDA 12.6이며, 환경을 바꿔 재학습했을 때 가중치 바이트까지 같다는 보장은 없다.

## 외부 평가와 해석

[`eval_sisfall.py`](scripts/eval_sisfall.py)는 `SisFall_dataset/S*/활동_피험자_반복.txt`의 파일명에서 피험자·라벨(`F`/`D`)을 읽는다. CSV 끝의 `;`를 제거하고 처음 3열에 `32/2¹³`(g), 다음 3열에 `4000/2¹⁶`(°/s)를 곱한다. 두 번째 가속도계는 사용하지 않는다. 축별 2표본 평균(200→100 Hz) 후 [동일 특징 창](scripts/train_eval.py#L20)을 25표본(0.25초)씩 이동한다. 짧은 파일은 첫 표본으로 왼쪽을 채운다. 파일 점수는 **모든 창의 sigmoid 중 최댓값**이고 `점수 > 0.5`만 양성이다. 시험셋으로 임계값을 조정하지 않는다. AUC는 양성 점수가 음성보다 큰 쌍의 비율에 동점의 절반을 더해 계산한다.

- 시험 피험자: **SA16–23 및 SE01–15**. SA01–15는 실측 비교 모델용이며 채택 모델은 그 기록으로도 학습하지 않았다. 원 자료의 동명 ID 5개(`D17_SE15_R01`–`R05`)는 원 평가처럼 정렬 후 마지막 경로만 센다. 분모는 창이나 사람 수가 아니라 **고유 기록 2,195건**이다.
- [원 결과 집계](results/reference.json)와 [동결 가중치 CPU 재실행](results/replay_cpu.json): **TP 390, FN 283, FP 5, TN 1,517**. Recall **57.95%**, FPR **0.329%**, ROC-AUC **0.95855**. 합성 내부 검증은 604/604 감지, 0/1,284 오경보다. 합성 100%를 실측 성능처럼 인용하면 안 된다.
- AUC는 점수 순위 지표다. 고정 임계값에서는 실제 시험 낙상 283건을 놓쳤고, 특히 보행 중 옆으로 미끄러지는 F03의 재현율은 22.2%다. 정지 출발 합성에 없는 동작이지만 그것이 미탐의 유일한 원인임을 입증한 것은 아니다. 파일별 FPR은 시간당 오경보율이 아니다.
- SisFall 신호를 모델 가중치 학습에 넣지 않았어도, 개발 중 실측 분포를 합성 물리 보정에 참고했고 이 시험군을 여러 비교 실험에서 확인했다. **미학습 피험자 벤치마크**이지 완전히 독립된 전향적 안전성 검증은 아니다. MPU6050 실사용, 장시간 모니터링, Jetson 지연시간 및 대화까지의 종단간 성능은 미검증이다.

## 실행 — 기본 경로는 무학습

명령은 `detector/`에서 실행한다. CPU 재검증 환경은 Python 3.14.7, NumPy 2.5.5, PyTorch 2.15.0+cpu, 생성기는 MuJoCo 4.0.1이었다. 아래 설치는 별도 환경을 준비할 때만 필요하다. 원 데이터, XML, 캐시는 배포물에 들어 있지 않다. 출처·해시·원본과의 차이는 [PROVENANCE.md](PROVENANCE.md) 참고.

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
sha256sum -c SHA256SUMS
python -m unittest discover -s tests -v   # XML 없이 6개 통과, 생성기 1개 건너뜀

# 공식 배포처에서 받은 SisFall.zip의 압축을 푼 뒤 경로를 지정한다.
python scripts/eval_sisfall.py /path/to/SisFall_dataset
```

**선택: 생성기 검증과 합성 데이터 재구축(학습 없음)**

```bash
python -m pip install -r requirements-sim.txt
mkdir -p assets runs
curl -fL 'https://raw.githubusercontent.com/google-deepmind/mujoco/395129bbd3976e9e6163ab5d3c480234f665d55a/model/humanoid/humanoid.xml' -o assets/humanoid.xml
printf '%s  %s\n' acc9167550bf20580717a551467e6e8d0d03d1916d24bbe4b734ce11afb99406 assets/humanoid.xml | sha256sum -c -
python -m unittest discover -s tests -v   # 8가지 동작을 포함해 7개 테스트
python scripts/synth_v1.py 300 --out runs/sim-v1
python scripts/check_synth.py runs/sim-v1/synth_v1.npz
```

이미 취득한 동일 XML은 `HUMANOID_XML`로 지정할 수 있다. 생성 결과 NPZ는 가변 길이 NumPy 객체 배열이므로 **직접 생성한 신뢰 가능한 파일만** 연다. `runs/sim-v1`은 새 경로여야 한다.

**참고 전용: 과거 학습 레시피** — 아래는 별도로 명시해 실행할 때만 새 학습을 한다. 이번 재현 검증에서는 실행하지 않았다. GPU를 쓰려면 앞에서 설정한 `CUDA_VISIBLE_DEVICES` 제한을 해제한다.

```bash
python scripts/train_synth.py runs/sim-v1/synth_v1.npz --out runs/retrained.pt
python scripts/eval_sisfall.py /path/to/SisFall_dataset --weights runs/retrained.pt
```

**검증 기록:** 단위 테스트 7개 통과([테스트 코드](tests/test_detector.py)). 각 동작의 첫 seed로 재생성한 8개 궤적은 보관된 원 합성 신호와 배열 원소가 전부 같았다. 전체 합성 창·분할을 다시 만들고 내부 검증 추론을 실행했다([집계](results/synth_replay.json)). SisFall은 원시 TXT부터 동결 가중치로 전체 시험군을 CPU에서 다시 추론했고 원 결과와 혼동행렬·재현율·FPR·AUC가 모두 일치했다. 전체 합성 기록의 재생성, 새 학습, 기기 배포는 이 검증에 포함하지 않았다.
