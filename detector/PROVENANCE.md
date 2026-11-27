# 출처와 변경 범위

**모델 식별자: `synth_v1` / 결정 기준일: 2026-11-27.** 11월 26일 확정한 안 A와 11월 27일자 프로젝트 MOC에서 순수 MuJoCo 기준치를 현행 CNN으로 확인했다. 파일 수정 날짜나 다른 실험의 높은 점수로 모델을 고르지 않았다. 원 작업 트리는 Git 스냅샷이 아니므로 임의의 커밋이나 릴리스 번호를 붙이지 않는다. 아래는 입력 원본의 SHA-256이며, 이 배포본 자체의 해시는 `SHA256SUMS`에 있다.

| 원본 | 배포본에서의 역할 | 원본 SHA-256 |
|---|---|---|
| `scripts/synth_v1.py` | 생성기·유효성 필터·seed 순서 유지. XML 위치와 해시 검사, 새 출력 디렉터리 인자 추가 | `b1cfadcfeab87de48fb363fc528c95501ff7a889fd5828a289fbc6cad3419884` |
| `scripts/train_eval.py` | 특징·창·Net·손실·학습 루프·예측·AUC를 가져옴. 합성 학습, 합성 검증, 외부 평가를 별도 명령으로 분리. 원본은 한 번 실행하면 실측 비교 모델까지 학습했지만 이 배포본은 그렇지 않다. 실측용 pickle 캐시 대신 TXT를 읽으며 같은 파일명 중 마지막 기록을 사용한다. | `e4628a457c2c876bb4bad7967ef4e5fd1c6d38d934a3de0598c69359f3b9ea1a` |
| `data/net_synth_v1.pt` | 동일 바이트의 합성 전용 PyTorch state_dict. 유일하게 동봉한 가중치 | `65e192e9b62cbd7458219029458fe7a2eb9811232def307d637bbe7402c592e4` |
| `data/eval_v1.json` | `results.synth_on_test` 및 저장 점수의 혼동행렬만 집계. 개별 예측 제외 | `fe5c67bfd1da168c7efd1d3b0d63e0ff6bf6832e34c92343a4aa09c4509c4b54` |
| `data/synth_v1.npz` / `data/synth_v1_meta.json` | 팀이 생성한 센서 기록 / seed·물리 매개변수. 코드로 재생성하며 원자료는 미포함 | `9563d0f5b98653859d13c522f089212e9e1b600f9699b012de63cf9b906efffa` / `52ea65c57dd7109abd6c14c6a5750932b7899af24e4d86bdb4cec5299934865a` |
| `SisFall.zip` | 평가에 사용한 배포본 식별용. 재배포하지 않음 | `7f0dd1583e39fee0f402a6855976eb12517f7eaecdacce871d5b3421793ad665` |

## 외부 입력
- **MuJoCo**: Google DeepMind 공식 저장소의 `model/humanoid/humanoid.xml`, 리비전 `395129bbd3976e9e6163ab5d3c480234f665d55a`. 공식 원본과 사용본의 SHA-256은 `acc9167550bf20580717a551467e6e8d0d03d1916d24bbe4b734ce11afb99406`으로 같다. XML·텍스처를 배포하지 않으며, 센서 추가는 실행 중 메모리에서 한다.
- **SisFall**: Sucerquia 등, *SisFall: A Fall and Movement Dataset*, Sensors (2017), DOI `10.3390/s17010198`. [논문의 Supplementary Materials 및 참고문헌 22](https://www.mdpi.com/1424-8220/17/1/198), [SISTEMIC 공식 배포처](https://sistemic.udea.edu.co/investigacion/proyectos/english-falls/?lang=en). 원 제공자의 이용 조건에 따라 텍스트 기록만 취득한다. 이번 검증은 기존 공식 배포 형식의 로컬 원시 TXT로 수행했으며 대학 서버의 새 다운로드는 확인하지 못했다.
- **HAPT·Isaac**: 현재 가중치 재현에 불필요하다. HAPT는 [UCI 공식 레코드 341](https://archive.ics.uci.edu/dataset/341) 참조. Isaac 모션은 조사·후속 계획일 뿐 다운로드할 필요가 없다.

개인 식별 정보, 인프라 설정, 실험 추적 로그, 원 노트, SisFall 원문·캐시·파일별 점수, NVIDIA 에셋, 제외 실험의 가중치는 배포하지 않는다.
