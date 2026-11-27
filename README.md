# 낙상 후 확인 대화 · 2조

이 저장소는 낙상 후보가 발생한 **뒤에 어떤 확인 질문이나 행동이 필요한지** 판단하는 대화 모델의 실험·재현 자료입니다. 모델은 센서 상태, 장치가 했던 질문, 사용자 음성의 STT 전사를 읽고 `evidence`, `call`, `next_prompt`를 JSON으로 제안합니다. 여기서는 실제 경보를 보내거나 장치를 구동하지 않습니다.

## 먼저 보기

- [Colab에서 노트북 열기](https://colab.research.google.com/github/yeounhyeok/aiot-fall-llm-harness/blob/main/notebooks/fall_dialogue_sft.ipynb) · [노트북 파일](notebooks/fall_dialogue_sft.ipynb) — 기본 모드는 CPU에서 자료 해시와 분할을 확인하고 기존 평가 결과를 표시합니다. 비공개 GitHub 접근 권한이 없으면 노트북 파일을 별도로 받아 Colab에 업로드하세요.
- [모델: 최종 LoRA + 중간 체크포인트 5개](https://huggingface.co/yeopeter1031/aiot-fall-qwen3.5-2b-lora-r1/tree/acd4d0a750731a600386d27c3969deb789dc6f1e) — 루트는 평가에 사용한 step 592입니다. 단독 모델이 아니므로 같은 저장소의 카드에 명시된 Qwen 원본과 함께 사용합니다.
- [티처 입력·출력, 선별 이력, 실제 SFT 분할](https://huggingface.co/datasets/yeopeter1031/aiot-fall-dialogue-teacher-10k-r1/tree/9082fca9a954148bba5deeee6cdb459b457414b2) — 학습에는 `sft/train.jsonl`만 사용합니다. 후보 보관 파일에는 검증 사례도 포함됩니다.
- [합성 음성 원본과 대화 ID별 매핑](https://huggingface.co/datasets/yeopeter1031/aiot-fall-dialogue-synthetic-audio-r1/tree/0b899dc20412071f3da36cea538b66e5b12ff144) — `row_audio.jsonl`에서 해당 턴의 ZIP과 WAV 경로를 확인할 수 있습니다. 실제 사람이나 낙상 현장의 녹음은 아닙니다.

## 진행 과정

1. **대화와 음성 만들기.** 장치의 질문과 사용자 대본을 만든 뒤 MMS-TTS-kor로 합성했습니다. 깨끗한 음성, 배경 소음, 끝부분 잘림 조건에서 Whisper-base로 전사했습니다. 학생에게 원 대본을 주지 않은 이유는 장치가 실제로 받는 입력이 전사문이기 때문입니다.
2. **라벨 생성·검토.** GPT-OSS-20B가 관측된 대화만 보고 다음 행동과 근거를 생성했습니다. 별도 블라인드 검토 결과, 함수·인자·후속 질문, 전사문에 근거가 있는지를 검사했습니다. 공유한 교사 라벨 및 TTS→STT 사례는 팀원이 직접 확인했습니다. 최종 1만 건을 사람이 전수 판정했다는 뜻은 아닙니다.
3. **학생 학습.** 필터 전 후보 14,680건 중 고유 대화 10,000건을 남겼습니다. 동일한 첫 관측 턴이 양쪽에 걸치지 않도록 학습 9,464건과 검증 536건으로 분리했습니다. Qwen3.5-2B의 언어층에 LoRA(rank 16, alpha 32)를 적용하고 티처의 최종 JSON과 EOS만 학습했습니다. 학습률 0.0001, 유효 배치 16, 1에폭(592스텝)입니다. 원 실험은 연구실 GPU에서 수행했습니다. Colab 노트북은 재현용이며 기존 결과의 실행 장소를 바꾸어 표기하지 않습니다.
4. **같은 조건에서 비교.** 미튜닝 Qwen3.5-2B와 최종 어댑터가 같은 536개 입력에 답했습니다(thinking off, greedy, 최대 160개 생성 토큰). 함수 이름·인자·후속 질문 경로까지 일치해야 정답입니다.

## 결과와 해석

| 검증 지표 | 미튜닝 | SFT |
|---|---:|---:|
| 행동 전체 일치 | 0/536 | 512/536 (95.5%) |
| 근거까지 포함한 JSON 전체 일치 | 0/536 | 422/536 (78.7%) |
| 교사 경보를 놓침 | 32/32 | 2/32 |
| 교사 비경보에 경보를 냄 | 0/504 | 0/504 |

항상 상태 재확인만 고르는 기준선도 444/536(82.8%)입니다. SFT는 이보다 68건을 더 맞혔지만, 직전 질문을 반복해야 하는 19건은 모두 상태 재확인으로 돌렸습니다. 미튜닝 0점은 이 직접 호출 조건에서 지정한 함수 형식을 맞히지 못했다는 뜻입니다. **이 점수는 합성 전사에서의 교사 일치도이지 실사용 낙상 감지 정확도가 아닙니다.** [집계 원본](https://huggingface.co/yeopeter1031/aiot-fall-qwen3.5-2b-lora-r1/resolve/acd4d0a750731a600386d27c3969deb789dc6f1e/evaluation.json)

## 재현 범위

노트북의 기본값 `RUN_TRAINING=False`는 추가 학습이나 모델 다운로드 없이 실행됩니다. 새 어댑터가 필요할 때만 BF16 지원 GPU에서 `True`로 설정하세요. 기존 점수는 새로 돌린 결과와 구분해 표시합니다. 이 저장소는 대화 모델 하네스이며, GGUF 배포본이나 실기기 통합 검증을 제공하지 않습니다.

## 초기 파일럿 (현재 10K 결과와 별개)

### 바로 실행 (저장소 루트, Python 3 표준 라이브러리)
```sh
python3 harness.py examples/cases.jsonl
python3 harness.py examples/cases.jsonl examples/mock_predictions.jsonl
python3 -m unittest -v
```
예시는 AI가 만든 모의 대화 3건으로, 팀이 검수한 학습·테스트셋이 아닙니다. 모의 예측 하나는 도움 요청을 잘못 해제하도록 작성했습니다. 정상 출력은 3건 중 정확히 일치 2건, 도움 요청 오해제 1건입니다. **모델 성능이 아닙니다.**

### 초기 파일럿의 사례 작성 규칙
1. `examples/cases.jsonl`을 참고해 `data/team_cases.jsonl`에 가명·모의 상황으로 원본을 한 줄씩 작성합니다. `id`는 사례별 고유값, `family`는 같은 원본과 변형이 공유하는 값입니다. 센서 상태가 모의인지 실측인지 표시합니다. 실제 이름, 연락처, 녹음은 커밋하지 않습니다.
2. 정답은 `raise_alert(help_requested)`, `ask_checkin(unclear_reply)`, `close_event(false_alarm)` 중 선택합니다. 이것은 첫 실험용 계약이며 의료 대응 기준이 아닙니다. 무응답 타이머는 LLM 채점 대상이 아닙니다.
3. 원본을 팀원이 교차 검토합니다. `source`에 작성 방식, 추가 필드 `reviewer`와 `review_note`에 검수자 별칭과 수정 이유를 남깁니다. 원본 단위로 `train`, `val`, `test`를 지정한 뒤 train에만 변형을 추가합니다. 모든 분할을 한 파일로 검사해야 가족 간 누출을 잡을 수 있습니다.
4. 모델에는 `sensor`, `question`, `reply`만 전달합니다. `gold`, ID, 출처는 프롬프트에 넣지 않습니다. 같은 테스트 사례로 모델별 `{"id":"…","call":{"name":"…","arguments":{"reason":"…"}}}` 예측 파일을 만들어 각각 채점합니다.
