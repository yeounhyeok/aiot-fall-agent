# 낙상 대응 LLM 하네스 · 2조

센서 모델의 판단, 직전 질문, 사용자 답변을 보고 **다음 함수 이름과 인자**를 맞혔는지 비교하는 오프라인 작업 공간입니다. 현재 호출은 데이터일 뿐 실행하지 않습니다. 학습 코드, 실제 센서·음성, 긴급 연락 기능은 없습니다.

## 바로 실행 (저장소 루트, Python 3 표준 라이브러리)
```sh
python3 harness.py examples/cases.jsonl
python3 harness.py examples/cases.jsonl examples/mock_predictions.jsonl
python3 -m unittest -v
```
예시는 AI가 만든 모의 대화 3건으로, 팀이 검수한 학습·테스트셋이 아닙니다. 모의 예측 하나는 도움 요청을 잘못 해제하도록 작성했습니다. 정상 출력은 3건 중 정확히 일치 2건, 도움 요청 오해제 1건입니다. **모델 성능이 아닙니다.**

## 팀원 첫 작업
1. `examples/cases.jsonl`을 참고해 `data/team_cases.jsonl`에 가명·모의 상황으로 원본을 한 줄씩 작성합니다. `id`는 사례별 고유값, `family`는 같은 원본과 변형이 공유하는 값입니다. 센서 상태가 모의인지 실측인지 표시합니다. 실제 이름, 연락처, 녹음은 커밋하지 않습니다.
2. 정답은 `raise_alert(help_requested)`, `ask_checkin(unclear_reply)`, `close_event(false_alarm)` 중 선택합니다. 이것은 첫 실험용 계약이며 의료 대응 기준이 아닙니다. 무응답 타이머는 LLM 채점 대상이 아닙니다.
3. 원본을 팀원이 교차 검토합니다. `source`에 작성 방식, 추가 필드 `reviewer`와 `review_note`에 검수자 별칭과 수정 이유를 남깁니다. 원본 단위로 `train`, `val`, `test`를 지정한 뒤 train에만 변형을 추가합니다. 모든 분할을 한 파일로 검사해야 가족 간 누출을 잡을 수 있습니다.
4. 모델에는 `sensor`, `question`, `reply`만 전달합니다. `gold`, ID, 출처는 프롬프트에 넣지 않습니다. 같은 테스트 사례로 모델별 `{"id":"…","call":{"name":"…","arguments":{"reason":"…"}}}` 예측 파일을 만들어 각각 채점합니다.
