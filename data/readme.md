## 📁 `data/` 디렉터리 설명

모두 말뭉치 평가 및 학습을 위한 데이터가 포함된 디렉터리입니다.

### ▶️ 주요 파일 설명

- `train.json`, `dev.json`, `test.json`
  - **[2025] 한국문화 질의응답(가 유형)**에서 제공한 원본 데이터셋

- `test_with_answers.csv`
  - `test.json`에 대해 GPT API를 활용해 임의로 정답을 생성한 평가셋
  - **majority voting** 기법으로 신뢰도 향상
  - **web search** 기능 포함하여 hallucination 방지
  - `web search` 기반 답변 vs `non-web search` 기반 답변이 다를 경우, 수작업으로 정답 교정

- `test_with_answers_서술형.csv`
  - `test_with_answers.csv` 중 서술형 문제만 추출한 파일

---

## 📂 `data/preprocessed/`

원본 데이터에서 **문제 형식**을 정제한 파일들 저장
- 예: `1\t 한국의 수도는?` → `1. 한국의 수도는?`

### 📂 `data/preprocessed/etc/`

- GPT API를 사용해 **web search 유무**에 따라 생성된 답변 데이터들 : `test_with_answers_websearch_1.json`, `test_with_answers_websearch_2.json`
- `test_with_answers_no_websearch.csv`: 검색 없이 생성된 답변들

---

## 🧪 GRPO 학습 관련 데이터

### ✅ `grpo_train.csv`
- GRPO 전체 학습 데이터셋

### ✅ `grpo_train_선다형.csv`, `grpo_train_단답형.csv`, `grpo_train_서술형.csv`
- 문제 유형별로 필터링된 학습셋

### ✅ `grpo_train_excluded_서술형.csv`
- 서술형 제외한 학습셋 (Rule-based 평가 어려움 때문)

### ✅ `grpo_train_excluded_서술형_midm.csv`
- 특정 모델 (예: KT/믿음) 기준 서술형 제외

### ✅ `grpo_train_excluded_서술형_midm_curriculum[_v2].csv`
- **Curriculum 기반 필터링 적용**
  - `v1`: 학습 전 8번 중 **정답률 0%**인 샘플 **제외**
  - `v2`: 정답률 0%인 샘플 **포함**
  - 참고: 정답률 **100%**인 샘플은 모두 제외됨

---

## 🎯 `sft_train.csv`
- Supervised Fine-tuning용 전체 학습셋

---

## 📄 기타 정리된 평가셋

- `test_with_answers.csv` : 전체 평가셋
- `test_with_answers_서술형.csv` : 서술형만 추출
