## 한눈에 보는 핵심 요약

이 레포지토리는 \*\*국립국어원 AI 말평(한국어 능력 평가)\*\*의 *한국문화 질의응답* 과제를 해결하기 위해 **2-단계 파이프라인(프롬프트 엔지니어링 →  GRPO 강화학습)** 을 구현한 토털 코드베이스입니다. 

**맞춤형 Reward Model** 및 **커리큘럼 샘플링** 기법까지 포함합니다. 
*설치 → 데이터 → 훈련 → 평가 → 추론* 과정

---

## 📂 프로젝트 구조

```
korean_culture_QA_2025
├── accelerate/                 # FSDP·DeepSpeed 설정
├── curriculum_data_sampling/   # 커리큘럼 데이터 필터링 스크립트
├── data/                       # 원본·전처리 데이터
├── examples/                   # 실험 shell script
├── initial_prompts/            # Zero-shot/CoT 프롬프트 실험
├── phase1_*                    # 프롬프트 기반 추론 실험
├── phase2_sft.py               # LoRA-SFT 학습
├── phase3_grpo*.py             # GRPO RL 학습 (단일/다중 GPU, FSDP)
├── reward_func.py              # 커스텀 RewardModel 함수
├── lora_model_merge.py         # LoRA 가중치 병합 유틸
├── result_graph.ipynb          # 평가 점수 시각화
└── install.sh / requirements.txt
```

## 🔧 설치 & 환경 구성

```bash
bash install.sh      # CUDA 12.x + PyTorch 2.3 기준
```


> **Tip** : GPU 4 장 이상이면 `accelerate config` 후 `accelerate launch phase3_grpo_6_fsdp.py`로 FSDP 학습을 실행할 수 있습니다.

## 📑 데이터 준비

1. **공식 Dev / Test** : 국립국어원 AI 말평 사이트에서 다운받아 `data/`에 저장합니다. ([국립국어원 언어정보나눔터])
2. `data_preprocessing.ipynb` : 데이터 전처리

## 🏗️ 3-단계 학습 파이프라인

| 단계                             | 스크립트               | 주요 기술                                                                     | 출력               |
| ------------------------------ | ------------------ | ------------------------------------------------------------------------- | ---------------- |
| **Phase 1** Prompt Engineering | `run_phase1.py`    | Zero/CoT/Choice-Shuffle 프롬프트 실험                                           | 초기 베이스라인         |
~~| **Phase 2** SFT                | `phase2_sft.py`    | **LoRA-8**(뒤 1/3 Attention Q,V), 16-bit, AdamW                            | SFT Adapter      |~~
| **Phase 3** RL                 | `phase3_grpo_*.py` | **GRPO**| 최종 RL checkpoint |

**Phase 3** RL은 선다형과 단단형에 적용