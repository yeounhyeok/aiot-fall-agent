# Fall Agent 유선 IMU 연결

`MPU6050 → I²C → Arduino Uno R3 → USB 시리얼 → WiredIMU → synth_v1 CNN → 상태 확인 대화`

이 어댑터는 낙상 **후보**를 대화 컨트롤러에 전달한다. CNN은 신고하거나 회복 여부를 판단하지 않는다. `snapshot()`은 후보가 발생한 창을 다시 보여 주는 대신, 질문에 답한 **뒤에 새로 관측한** 움직임을 반환한다. 센서 오류는 ‘움직임 없음’이 아니다.

## 배선과 펌웨어

1. 전원을 끄고 Uno R3의 **SDA(A4) ↔ SDA**, **SCL(A5) ↔ SCL**, **GND ↔ GND**를 연결한다. 센서 전원과 I²C 풀업 전압은 사용하는 브레이크아웃 회로도에 맞춘다. 3.3 V 전용 센서에 5 V 신호를 직접 걸지 않는다. 5 V Uno와 3.3 V 모듈 조합은 모듈에 레벨 변환기가 없으면 양방향 I²C 레벨 변환기를 사용한다. 다른 보드는 A4/A5가 아니라 해당 보드의 SDA/SCL 표기를 따른다.
2. AD0를 Low로 연결하면 주소는 `0x68`이다. High라면 스케치의 `ADDRESS`를 `0x69`로 변경한다. INT 핀은 연결하지 않는다. 센서를 몸에 고정할 때는 학습 위치인 골반 부근을 기준으로 하며 케이블을 걸림 없이 고정한다. 사람에게 실제 낙상을 시키지 않는다.
3. USB 데이터 케이블로 연결한다. Arduino IDE에서 `firmware/wired_imu/wired_imu.ino`와 보드·포트를 선택한다. 아래는 Arduino AVR 코어가 설치된 Uno R3의 CLI 예시다. 업로드는 기존 스케치를 덮어쓰므로 보드 소유자의 확인 후 실행한다.

```sh
arduino-cli board list
PORT=/dev/ttyACM0       # 실제 포트로 교체
arduino-cli compile --fqbn arduino:avr:uno firmware/wired_imu
arduino-cli upload -p "$PORT" --fqbn arduino:avr:uno firmware/wired_imu
```

## 호스트 실행

Linux 호스트에 USB로 연결한다. 호스트 GPIO/I²C 설정은 필요 없다. 아래 명령은 `runtime/`과 `firmware/`가 있는 디렉터리에서 실행한다. 기존 체크포인트와 `detector/scripts/train_eval.py`가 필요하다. 모델 다운로드·재학습·모의 점수 대체 경로는 없다. 검증된 Python 환경이 있으면 재사용한다. 새 환경을 구성하는 경우 해당 보드/Python과 호환되는 CPU PyTorch 및 NumPy를 준비하고 같은 환경에 `pyserial==3.5`를 설치한다. 기준 버전은 기존 `detector/requirements.txt`에 있다.

```sh
export FALL_DETECTOR_DIR=./detector   # 별도 체크아웃이면 해당 detector/ 경로
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m runtime.imu --port "$PORT" --detector-dir "$FALL_DETECTOR_DIR" --observe
```

정지시킨 보드에서 `최근 0.5초 측정: 최대 가속도 크기 … g, 평균 각속도 크기 … °/s.`를 확인한다. 정지 시 가속도 크기는 중력을 포함해 대략 1 g이고 각속도 크기는 작아야 한다. `--observe`를 빼면 첫 후보를 기다리고 새 관측 한 번을 출력한 뒤 종료한다. **후보가 없는 동안 조용한 것이 정상이다.** 운영에서는 단일 소비자가 컨텍스트를 소유한다.

```python
from runtime.controller import monitor
from runtime.imu import WiredIMU, SensorError

# io: 컨트롤러의 음성·대화 I/O 구현. alerts=False는 외부 알림을 비활성화한다.
try:
    with WiredIMU(port=port, detector_dir=detector_dir) as sensor:
        monitor(sensor, io, one=True, alerts=False)
except (SensorError, OSError) as exc:
    print(f'센서 사용 불가: {exc}')  # 안전/회복으로 처리하지 않는다
```

- `wait_candidate()`는 새 후보에서만 반환한다. 100 Hz 연속 300표본부터 25표본마다 기존 `feats`와 `Net`으로 추론한다. 특징은 `|a|`, `|ω|/100`, 창 첫 50표본 평균 대비 최근 최대 25표본 평균 가속도의 각도다. 단일 CPU 스레드, `eval()` 및 `inference_mode()`를 사용한다. 판정은 `sigmoid(logit) > 0.5`이며 첫 3초는 패딩하지 않는다. 체크포인트 SHA-256은 `65e192e9b62cbd7458219029458fe7a2eb9811232def307d637bbe7402c592e4`로 고정한다.
- 대화 중에도 리더 스레드가 최대 300표본만 보존한다. `snapshot()`은 호출 후 연속 50표본의 최대 가속도 크기와 평균 각속도 크기를 한국어 문자열로 반환한다. 이전 점수를 진단 문구로 변환하지 않는다.
- `rearm()`은 대화 중 후보를 버린다. 3초가 지난 뒤 `p ≤ 0.35`를 한 번 관측하고, 이후 다시 `p > 0.5`가 되어야 다음 대화를 연다. 이는 중복 방지 규칙이며 검증된 임상 임계값이 아니다.

## 선로 형식과 장애 처리

펌웨어는 MPU6050에서 DLPF 설정 3, 분주기 9로 100 Hz 데이터를 읽는다. DATA_RDY마다 `0x3B`에서 14바이트를 한 번에 받아 **115200 baud, 8N1**로 전송한다. USB 한 줄은 `IMU1,<seq>,<us>,<28자리 hex>\r\n`이다. `seq`는 0부터 증가하는 uint32이고 `us`는 보드의 `micros()`다. 호스트 도착 시각으로 샘플링 주기를 추측하지 않는다.

예: `IMU1,42,1234,f800000008007fff00a4ff5c0000` → 가속도 `(-1, 0, 1) g`, 각속도 `(10, -10, 0) °/s`. 페이로드는 big-endian int16 `ax, ay, az, 온도, gx, gy, gz` 순서다. 온도는 건너뛴다. ±16 g 설정에서 `2048 LSB/g`, ±2000 °/s 설정에서 `16.4 LSB/(°/s)`로 나눈다. SI 단위나 소수점 CSV를 보내는 스케치와 호환되지 않는다.

- 누락·중복·형식 오류 또는 표본 간격이 8–12 ms를 벗어나면 창을 다시 채운다. 카운터 순환은 허용하되 300표본 창의 경과 시간이 공칭값에서 3%를 넘게 벗어나면 추론하지 않는다.
- `ERR,I2C`이면 전원·주소·배선·칩 종류를 확인한다. 시작 후 6초 동안 유효 프레임이 없거나 수신 중 0.75초가 지나면 오류다. 장기간 온전한 창이 없거나 시리얼 버퍼가 1024바이트를 넘으면 과거 데이터를 실시간 사건으로 전달하지 않고 중단한다. 재연결 후에는 새 컨텍스트를 연다.
- 포트 열기에 실패하면 USB 데이터 케이블, 포트명과 사용자 접근 권한(배포판에 따라 `dialout`)을 확인하고 IDE의 시리얼 모니터 등 다른 소비자를 닫는다. 체크포인트 해시 오류는 일치하는 프로젝트 배포본을 지정해 해결한다. 임계값 변경으로 우회하지 않는다.

## 하드웨어 없는 검증과 범위

`python -m unittest -v runtime.test_imu`는 레지스터 부호·축·단위, 잘못된 프레임, 카운터 순환, 누락과 샘플률, 재무장, Linux 가상 시리얼 포트(PTY)의 후보·후속 관측·수신 중단·분리를 확인한다. `FALL_DETECTOR_DIR`로 원본 감지기 위치를 지정할 수 있다. 별도로 보유한 원시 SisFall 기록 두 개도 재현하려면 `SISFALL_ROOT`를 `SA16/`이 있는 데이터셋 폴더로 지정한다. 공개 데이터 원문은 동봉하지 않는다.

`tests/fixtures/imu/{topple,stand}.npy`는 팀의 기존 `synth_v1.npz`(SHA-256 `9563d0f5b98653859d13c522f089212e9e1b600f9699b012de63cf9b906efffa`)의 `X[0]`(seed 0)과 `X[900]`(seed 300000)을 숫자 배열로 추출한 것이다. 각각 `(2100,6)` float32, 500 Hz이며 테스트에서 연속 다섯 표본을 평균해 100 Hz로 만든다. 새 시뮬레이션이나 학습은 하지 않는다.

기존 CPU 체크포인트로 다섯 창씩 재생했을 때 최대 점수는 `topple` 0.9999995, `stand` 0.0000431이었다. 별도의 SisFall 실측 재생에서는 선택한 낙상 기록이 0.380829로 **미탐**이었고 일상 기록은 0.000136이었다. 이 소수의 사례는 연결 경로를 확인한 것이지 현장 정확도 측정이 아니다. Uno 대상 펌웨어는 컴파일·링크했지만 보드 업로드, 실제 MPU6050 신호, Jetson 지연시간, 음성·알림까지의 종단간 동작은 검증하지 않았다.
