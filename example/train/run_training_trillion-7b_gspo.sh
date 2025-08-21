#!/usr/bin/env bash
set -e
ACC_CFG="/workspace/korean_culture_QA_2025/accelerate/fsdp_v1_config.yaml"   # accelerate config 결과 저장해둔 파일

# export CUDA_VISIBLE_DEVICES=0,1
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29500
# export NCCL_P2P_DISABLE=0
# export NCCL_IB_DISABLE=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_BLOCKING_WAIT=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# # SHM 작은 환경이면 일단 켭니다 (재기동 전 임시 회피)
# export NCCL_SHM_DISABLE=1

system_prompt="당신은 한국의 문화와 관련된 문제를 전문적으로 풀이해주는 문제 해설가입니다.  
사용자가 입력한 문제에 대해 단계별로 차근차근(step by step) 설명하여 **문제 해설**과 **정답**을 제시하세요.  

답변 형식은 반드시 다음과 같이 작성하세요:  
문제 해설: (문제에 대한 배경, 개념 설명, 선택지 또는 정답 후보 분석 등 단계별로 논리적인 추론 과정을 포함)  
정답: (정답만 간결하게 작성)

※ 정답 작성 형식 안내  
- 선다형 문제일 경우: **정답 번호**만 작성하세요.  
- 단답형 문제일 경우:  
    - 정답이 1개인 경우: 괄호, 한자 없이 **한글** 또는 **영어**를 사용하세요. (예: 사과 또는 apple)  
    - 정답이 여러 개인 경우: 쉼표(,)로 구분해 나열하세요. (예: 사과, 배)  
    - 순서 배열 문제인 경우: '-'로 구분해 정확한 순서를 유지해 나열하세요. (예: ㄱ-ㄴ-ㄷ-ㄹ)"

user_prompt="한국의 문화와 관련된 아래 문제를 단계별로 자세히 해설해주고, 마지막에 정답을 작성해줘.  

키워드: {topic_keyword}  
문제 유형: {question_type}  
문제: {question}"

answer_tag="정답:"


nohup accelerate launch \
  --config_file "$ACC_CFG" \
  --num_processes 2 \
  -m scripts.train.phase3_grpo_6_fft \
  --model "trillionlabs/Tri-7B" \
  --temperature 1.0 \
  --epochs 10 \
  --epsilon 0.0003 \
  --epsilon_high 0.0004 \
  --loss_type "grpo" \
  --importance_sampling_level "sequence" \
  --system_prompt "$system_prompt" \
  --prompt_template "$user_prompt" \
  --solution_start "$answer_tag" \
  --train_data "/workspace/korean_culture_QA_2025/data/preprocessed/grpo_train_excluded_서술형_trillion_curriculum_v2.csv" \
  --valid_data "/workspace/korean_culture_QA_2025/data/preprocessed/original_dev_excluded_서술형.csv" \
  --save_name "original_train_선다형_단답형_v2_prompt2_gspo_fft" \
  > train.log
