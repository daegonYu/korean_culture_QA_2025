
# **korean\_culture\_QA\_2025**

“국립국어원 AI 말평(한국어 능력 평가)”의 한국문화 QA 과제를 해결하기 위한 **2단계 파이프라인** 구현 코드베이스입니다.
프롬프트 엔지니어링부터 GRPO 기반 강화학습까지 포함한 토탈 솔루션을 제공합니다.

---

## 주요 내용

* **프롬프트 엔지니어링 (Phase 1)**: Zero-shot, CoT 기반 실험을 통해 최적의 프롬프트 전략 탐색
* **감독 학습 (Phase 2)**: 수집된 고품질 응답을 기반으로 SFT(LoRA) 방식으로 모델 미세조정
* **강화학습 (Phase 3)**: 그룹 상대 정책 최적화(GRPO)를 통해 의도한 품질 향상 — 단일/다중 GPU, FSDP 버전 지원
* **커리큘럼 데이터 샘플링**: 8번 추론하여 모두 맞힌 데이터와 모두 틀린 데이터 제외

---

## 프로젝트 구조

```
korean_culture_QA_2025
├─ accelerate/ # FSDP·DeepSpeed 설정 템플릿
├─ data/ # (사용자) 원본/전처리 데이터 경로
├─ example/ # 실행 예시용 스크립트/명령 모음
├─ initial_prompts/ # Zero-shot/CoT 등 프롬프트 실험 자료
├─ results/ # 평가/로그/그래프 산출물
├─ scripts/ # 유틸/실행 스크립트
├─ utils/ # 보조 함수
├─ phase2_sft.py # (옵션) LoRA SFT 학습 스크립트
├─ phase3_grpo_*.py # GRPO RL 학습 스크립트(단일·다중 GPU)
├─ reward_func.py # 커스텀 보상 함수 모음
├─ lora_model_merge.py # LoRA 가중치 병합 유틸
├─ install.sh # 의존성 설치 스크립트
├─ requirements.txt # Python 의존성
└─ result_graph.ipynb # 결과 시각화 노트북
```

---

## 설치 및 환경 구성

```bash

## ⚙️ 환경 준비

### 1) 사전 요구사항
- OS: Linux (권장)
- Python: 3.10+
- CUDA: 12.x / PyTorch: 2.3.x (프로젝트 스크립트 기준 권장)

### 2) 설치
```bash
# 1) 레포 클론
git clone https://github.com/daegonYu/korean_culture_QA_2025.git
cd korean_culture_QA_2025

# 2) 가상환경 (예: conda)
conda create -n kcqa python=3.10 -y
conda activate kcqa

# 3) 의존성 설치
bash install.sh
# 또는
pip install -r requirements.txt
```

GPU가 4장 이상이면, 다음과 같이 FSDP 기반 GRPO 학습 실행이 가능합니다:

```bash
accelerate config
accelerate launch phase3_grpo_6_fsdp.py
```

---

## 데이터 준비

1. **국립국어원 AI 말평** 공식 Dev/Test 데이터를 다운로드하여 `data/` 폴더에 저장
2. `data_preprocessing.ipynb`를 통해 데이터 전처리 수행

---

## 파이프라인 요약

| 단계      | 스크립트               | 주요 내용                        |
| ------- | ------------------ | ---------------------------- |
| Phase 1 | `run_phase1.py`    | 다양한 프롬프트 전략으로 Baseline 응답 생성 |
| Phase 2 | `phase2_sft.py`    | SFT 기반 미세조정 (LoRA)     |
| Phase 3 | `phase3_grpo_*.py` | GRPO 기반 RL 학습, FSDP 지원 포함    |

* **Phase 2**: 서술형 응답 SFT 
* **Phase 3**: 선다형 + 단답형 QA 모두에 적용

---

## 도움말 및 확장

* **실험 자동화**: `examples/` 폴더에서 다양한 실험 시나리오를 확인할 수 있습니다.
* **시각화 결과**: `result_graph.ipynb`을 통해 성능 비교와 추세 파악이 가능합니다.

---


