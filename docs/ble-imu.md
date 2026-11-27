# UNO R4 WiFi 착용 센서 → Jetson BLE

`보조배터리 → UNO R4 WiFi + MPU6050 → BLE → Jetson CNN` · `마이크 / 스피커 → Jetson 유선`

별도 공유기나 인터넷 없이 센서 신호만 전송합니다. Discord 알림을 사용할 때만 외부 네트워크가 필요합니다. 무선 음성 장치와 클라우드 추론은 사용하지 않습니다.

1. 전원을 끄고 MPU6050의 SDA/SCL을 UNO의 **SDA/SCL 표기 핀**에, GND를 공통 접지에 연결합니다. 센서 모듈 전원 및 풀업 전압을 회로도와 대조하세요. 3.3 V 전용 센서는 5 V 신호선에 직결하지 않습니다. AD0 Low(0x68)가 기본입니다.
2. PC의 Arduino IDE에서 **Arduino UNO R4 WiFi** 보드와 `ArduinoBLE` 라이브러리를 선택하고 [BLE 스케치](../firmware/ble_imu/ble_imu.ino)를 업로드합니다. R3용 스케치와 구분하세요. 보드 연결 펌웨어는 ArduinoBLE 호환 버전이 필요합니다. PC에서 분리한 뒤 USB-C 보조배터리로 전원을 공급하면 시리얼 터미널을 기다리지 않고 `Fall-IMU`로 광고합니다.
3. Jetson에 BlueZ와 에이전트 Python 환경의 `bleak` 패키지가 필요합니다(`python3 -m pip install bleak`). `bluetoothctl list`에 컨트롤러가 있는지 확인하고, `bluetoothctl`에서 `power on`, `scan on`으로 본인 보드의 `Fall-IMU` 주소를 확인한 뒤 `scan off`, `quit`를 실행합니다. 보드 이름만 보고 자동 연결하지 않습니다. 주소 선택은 암호 인증이 아니므로 이 버전은 통제된 시연 환경용입니다.
4. [로컬 대화 준비](local-dialogue.md)에 따라 기존 마이크·스피커와 WAV, Whisper, GGUF 서버를 준비한 뒤 저장소 루트에서 실행합니다. 주소와 경로는 실제 값으로 바꾸세요.

```sh
python3 -m runtime live --ble-address "$UNO_BLE_ADDRESS" --asr-dir /models/whisper-base
```

**데이터 계약:** 목표 측정 주기는 100 Hz입니다. 알림 1건 = 20바이트로, 표본 순번·보드 `micros()`(각 uint32 little-endian) + 가속도 XYZ·각속도 XYZ(각 int16 big-endian)입니다. CNN은 USB와 동일한 단위·전처리·3초 창을 사용합니다. 빠진 순번과 잘못된 표본 간격을 건너 이어 붙이지 않습니다. 단절·장기 지연은 ‘정상/회복’으로 판정하지 않고 오류로 종료합니다. 무선 상태를 복구한 뒤 프로그램을 다시 시작하세요.

**USB 백업:** 같은 R4 스케치에서 BLE 구독이 없고 PC/Jetson 시리얼 포트가 열려 있으면 USB로 `IMU1` 프레임을 출력합니다. BLE 프로그램을 종료하고 USB 데이터 케이블을 Jetson에 연결한 뒤 `--ble-address` 대신 `--port /dev/ttyACM0`(실제 포트로 교체)를 사용하세요. BLE 초기화 장애 시에는 기존 유선 스케치를 별도로 사용할 수 있습니다. 실제 보드 펌웨어 업로드, BLE 링크의 지속 처리율, 착용 상태에서의 동작은 검증 전입니다.
