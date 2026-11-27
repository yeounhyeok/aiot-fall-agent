# 장치용 안내 음성 재생

Python 3과 ALSA의 `aplay`로 합성 안내 WAV를 오프라인에서 재생합니다. 장치에는 음성 복제 모델이나 참조 녹음이 필요하지 않습니다. 개인 음성 팩은 별도로 전달하며, 팩의 `device_audio/wav/`를 이 디렉터리에 두세요.

```sh
python3 device_audio/play.py initial_check --dry-run
python3 device_audio/play.py initial_check
python3 device_audio/play.py repeat_last --previous offer_help
```

`initial_check`는 첫 질문, `clarify_status`는 상태 재확인, `offer_help`는 도움 제안, `close_event`는 확인 종료입니다. `raise_alert`는 도움 요청 확인 멘트만 재생합니다. Discord 발송은 별도 알림 함수가 담당합니다. `repeat_last`에는 마지막으로 **실제 재생한 질문 ID**를 넘기세요. 출력 장치 선택은 `--device` 옵션을 사용합니다. 실제 보드·스피커 재생은 별도로 확인해야 합니다.
