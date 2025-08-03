#!/usr/bin/env bash
set -e


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

user_prompt="아래 문제를 단계별로 자세히 해설해주고, 마지막에 정답을 작성해줘.  

키워드: {topic_keyword}  
문제 유형: {question_type}  
문제: {question}"

answer_tag="정답:"


nohup accelerate launch --config_file accelerate/fsdp_config.yaml phase3_grpo_6_fsdp.py \
  --model "skt/A.X-4.0-Light" \
  --temperature 1.1 \
  --epochs 15 \
  --system_prompt "$system_prompt" \
  --prompt_template "$user_prompt" \
  --solution_start "$answer_tag" \
  --data_path "/workspace/korean_culture_QA_2025/data/preprocessed/grpo_train_excluded_서술형_skt_curriculum.csv" \
  --save_name "curri_선다형_단답형_v1_prompt2_FFT" \
  --vllm_gpu_memory_utilization 0.6


paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v6_A.X-4.0-Light_curri_선다형_단답형_v1_prompt2_FFT"
)

for path in "${paths[@]}"; do
    echo "🔍 상위 경로: $path"
    find "$path" -mindepth 1 -type d | while read -r model; do
        echo "🔍 현재 경로: $model"
        nohup python run_phase1.py --model "$model" --use_test --use_wandb \
        --system_prompt "$system_prompt" --user_prompt "$user_prompt" --answer_tag "$answer_tag"

        dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
        checkpoint=$(basename "$model")            # "checkpoint-112"
        model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

        echo "Model name: $model_name"

        json_path="results/phase1_${model_name}_test_outputs.json"
        echo "Scoring: $json_path"
        python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
    done
done
