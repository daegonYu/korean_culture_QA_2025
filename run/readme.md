# 🛠️ 프로젝트 실행 가이드

이 저장소는 **모델 학습(Train)**, **개발(dev) 셋 평가**, **테스트(test) 셋 평가**,  
그리고 **이미 생성된 결과에 점수만 매기는 작업(Scoring-only)** 을 손쉽게 실행할 수 있도록 디렉터리별 스크립트를 제공합니다.

---


## 🚀 사용 방법


### 1. 모델 학습 (Train)

```bash
# 기본 하이퍼파라미터로 학습
./train/run_training.sh

# 여러 설정을 한 번에 돌리고 싶을 때
./train/run_training_all.sh
````

### 2. Dev 셋 평가

```bash
./dev/run_dev.sh
```

### 3. Test 셋 평가

```bash
# 단일 설정
./test/run_test.sh

# 사전에 정의된 모든 설정으로 일괄 테스트
./test/run_test_all.sh
```

### 4. Scoring-only 모드

모델이 이미 생성해 둔 출력(JSON/CSV 등)에 대해 **정답과의 일치 정도**만 산출하고 싶을 때 사용합니다.

```bash
# 단일 결과 파일 스코어링
./scoring_only/score_only.sh

# 여러 결과 파일 일괄 스코어링
./scoring_only/score_only_all.sh
```

> 내부적으로 `score_only_answer.py`가 불려서, reference file 대비 정확도·F1 등 지표를 계산합니다.

---

## 📝 참고

* **로그 & 체크포인트**: 실행 결과(모델 가중치, 로그)는 기본적으로 `results/` 하위에 저장되며, 필요에 따라 스크립트에서 디렉터리를 지정하세요.

