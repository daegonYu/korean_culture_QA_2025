#!/usr/bin/env bash
set -e


# system_prompt="당신은 한국의 문화와 관련된 문제를 전문적으로 풀이해주는 문제 해설가입니다.  
# 사용자가 입력한 문제에 대해 정확하고 친절하게 **문제 해설**과 **정답**을 제시하세요.  
# 답변 형식은 반드시 다음과 같이 작성하세요:  
# 문제 해설: ...  
# 정답: ...

# ※ 정답 작성 형식 안내  
# - 선다형 문제일 경우: **정답 번호**를 작성하세요.  
# - 단답형 문제일 경우:  
#     - 정답이 1개인 경우: 괄호, 한자 없이 **한글** 또는 **영어**를 사용하세요. (예: 사과 또는 apple)  
#     - 정답이 여러 개인 경우: 쉼표(,)로 구분해 나열하세요. (예: 사과, 배)  
#     - 순서 배열 문제인 경우: '-'로 구분해 정확한 순서를 유지해 나열하세요. (예: ㄱ-ㄴ-ㄷ-ㄹ)"

# user_prompt="아래 문제를 해설해주고 정답을 알려줘.\n반드시 문제 해설 다음에 정답을 작성해줘.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n문제 유형: {question_type}\n\n문제: {question}\n\n답변:"

# # nohup python phase3_grpo_4.py \
# # --model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum.csv" --system_prompt "$system_prompt"

# # nohup python phase3_grpo_4.py \
# # --model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum_v2.csv" --system_prompt "$system_prompt"



# # 여러 경로를 배열로 선언
# paths=(
#     # "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v1"
#     # "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v2"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v1/checkpoint-4"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v1/checkpoint-8"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v1/checkpoint-12"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v1/checkpoint-16"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v1/checkpoint-20"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v1/checkpoint-24"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v1/checkpoint-28"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v2/checkpoint-6"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v2/checkpoint-12"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v2/checkpoint-18"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v2/checkpoint-24"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v2/checkpoint-30"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v2/checkpoint-36"
# )
# answer_tag="정답:"

# # 각 경로에 대해 하위 디렉토리 순회
# for model in "${paths[@]}"; do
#     # echo "🔍 상위 경로: $path"
#     # find "$path" -mindepth 1 -type d | while read -r model; do
#     echo "🔍 현재 경로: $model"
#     nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
#     --system_prompt "$system_prompt" --user_prompt "$user_prompt" --answer_tag "$answer_tag" >> logs/test.log

#     dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
#     checkpoint=$(basename "$model")            # "checkpoint-112"
#     model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

#     echo "Model name: $model_name"

#     # Phase 2: Scoring
#     json_path="results/phase1_${model_name}_test_outputs.json"
#     echo "Scoring: $json_path"
#     python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
#     # done
# done

####

# system_prompt="당신은 한국의 문화와 관련된 문제를 전문적으로 풀이해주는 문제 해설가입니다.  
# 사용자가 입력한 문제에 대해 정확하고 친절하게 **문제 해설**과 **정답**을 제시하세요.  
# 답변 형식은 반드시 다음과 같이 작성하세요:  
# 문제 해설: ...  
# 정답: ...

# ※ 정답 작성 형식 안내  
# - 선다형 문제일 경우: **정답 번호**를 작성하세요.  
# - 단답형 문제일 경우:  
#     - 정답이 1개인 경우: 괄호, 한자 없이 **한글** 또는 **영어**를 사용하세요. (예: 사과 또는 apple)  
#     - 정답이 여러 개인 경우: 쉼표(,)로 구분해 나열하세요. (예: 사과, 배)  
#     - 순서 배열 문제인 경우: '-'로 구분해 정확한 순서를 유지해 나열하세요. (예: ㄱ-ㄴ-ㄷ-ㄹ)"

# user_prompt="아래 문제를 해설해주고 정답을 알려줘.\n반드시 문제 해설 다음에 정답을 작성해줘.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n문제 유형: {question_type}\n\n문제: {question}\n\n답변:"

# nohup python phase3_grpo_4.py \
# --model "kakaocorp/kanana-1.5-8b-instruct-2505" --epochs 5 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum.csv" --system_prompt "$system_prompt"

# nohup python phase3_grpo_4.py \
# --model "kakaocorp/kanana-1.5-8b-instruct-2505" --epochs 5 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum_v2.csv" --system_prompt "$system_prompt"

# paths=(
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_kanana-1.5-8b-instruct-2505_curri_선다형_단답형_v1"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_kanana-1.5-8b-instruct-2505_curri_선다형_단답형_v2"
# )
# answer_tag="정답:"

# for path in "${paths[@]}"; do
#     echo "🔍 상위 경로: $path"
#     find "$path" -mindepth 1 -type d | while read -r model; do
#         echo "🔍 현재 경로: $model"
#         nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
#         --system_prompt "$system_prompt" --user_prompt "$user_prompt" --answer_tag "$answer_tag" >> logs/test.log

#         dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
#         checkpoint=$(basename "$model")            # "checkpoint-112"
#         model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

#         echo "Model name: $model_name"

#         json_path="results/phase1_${model_name}_test_outputs.json"
#         echo "Scoring: $json_path"
#         python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
#     done
# done


# nohup python phase3_grpo_4.py \
# --model "skt/A.X-4.0-Light" --epochs 5 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_lr" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum.csv" --system_prompt "$system_prompt"

# nohup python phase3_grpo_4.py \
# --model "skt/A.X-4.0-Light" --epochs 5 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_lr" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum_v2.csv" --system_prompt "$system_prompt"

# paths=(
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_X-4.0-Light_curri_선다형_단답형_v1_lr"
#     "/workspace/korean_culture_QA_2025/models/grpo_v3_X-4.0-Light_curri_선다형_단답형_v2_lr"
# )
# answer_tag="정답:"

# for path in "${paths[@]}"; do
#     echo "🔍 상위 경로: $path"
#     find "$path" -mindepth 1 -type d | while read -r model; do
#         echo "🔍 현재 경로: $model"
#         nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
#         --system_prompt "$system_prompt" --user_prompt "$user_prompt" --answer_tag "$answer_tag" >> logs/test.log

#         dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
#         checkpoint=$(basename "$model")            # "checkpoint-112"
#         model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

#         echo "Model name: $model_name"

#         json_path="results/phase1_${model_name}_test_outputs.json"
#         echo "Scoring: $json_path"
#         python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
#     done
# done


# system_prompt="당신은 한국의 문화와 관련된 문제를 전문적으로 풀이해주는 문제 해설가입니다.  
# 사용자가 입력한 문제에 대해 단계별로 차근차근(step by step) 설명하여 **문제 해설**과 **정답**을 제시하세요.  

# 답변 형식은 반드시 다음과 같이 작성하세요:  
# 문제 해설: (문제에 대한 배경, 개념 설명, 선택지 또는 정답 후보 분석 등 단계별로 논리적인 추론 과정을 포함)  
# 정답: (정답만 간결하게 작성)

# ※ 정답 작성 형식 안내  
# - 선다형 문제일 경우: **정답 번호**만 작성하세요.  
# - 단답형 문제일 경우:  
#     - 정답이 1개인 경우: 괄호, 한자 없이 **한글** 또는 **영어**를 사용하세요. (예: 사과 또는 apple)  
#     - 정답이 여러 개인 경우: 쉼표(,)로 구분해 나열하세요. (예: 사과, 배)  
#     - 순서 배열 문제인 경우: '-'로 구분해 정확한 순서를 유지해 나열하세요. (예: ㄱ-ㄴ-ㄷ-ㄹ)"

# user_prompt="아래 문제를 단계별로 자세히 해설해주고, 마지막에 정답을 작성해줘.  

# 키워드: {topic_keyword}  
# 문제 유형: {question_type}  
# 문제: {question}"


# nohup python phase3_grpo_5.py \
# --model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum.csv" --system_prompt "$system_prompt"

# nohup python phase3_grpo_5.py \
# --model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum_v2.csv" --system_prompt "$system_prompt"

# paths=(
#     "/workspace/korean_culture_QA_2025/models/grpo_v5_A.X-4.0-Light_curri_선다형_단답형_v1_prompt2"
#     "/workspace/korean_culture_QA_2025/models/grpo_v5_A.X-4.0-Light_curri_선다형_단답형_v2_prompt2"
# )
# answer_tag="정답:"

# for path in "${paths[@]}"; do
#     echo "🔍 상위 경로: $path"
#     find "$path" -mindepth 1 -type d | while read -r model; do
#         echo "🔍 현재 경로: $model"
#         nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
#         --system_prompt "$system_prompt" --user_prompt "$user_prompt" --answer_tag "$answer_tag" >> logs/test.log

#         dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
#         checkpoint=$(basename "$model")            # "checkpoint-112"
#         model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

#         echo "Model name: $model_name"

#         json_path="results/phase1_${model_name}_test_outputs.json"
#         echo "Scoring: $json_path"
#         python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
#     done
# done


# nohup python phase3_grpo_5.py \
# --model "K-intelligence/Midm-2.0-Base-Instruct" --epochs 4 --temperature 1.0 --lora_rank 8 --lora_alpha 8 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_midm_curriculum.csv" --system_prompt "$system_prompt"

# nohup python phase3_grpo_5.py \
# --model "K-intelligence/Midm-2.0-Base-Instruct" --epochs 4 --temperature 1.0 --lora_rank 8 --lora_alpha 8 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_midm_curriculum_v2.csv" --system_prompt "$system_prompt"

# paths=(
#     "/workspace/korean_culture_QA_2025/models/grpo_v5_Midm-2.0-Base-Instruct_curri_선다형_단답형_v1_prompt2"
#     "/workspace/korean_culture_QA_2025/models/grpo_v5_Midm-2.0-Base-Instruct_curri_선다형_단답형_v2_prompt2"
# )
# answer_tag="정답:"

# for path in "${paths[@]}"; do
#     echo "🔍 상위 경로: $path"
#     find "$path" -mindepth 1 -type d | while read -r model; do
#         echo "🔍 현재 경로: $model"
#         nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
#         --system_prompt "$system_prompt" --user_prompt "$user_prompt" --answer_tag "$answer_tag" >> logs/test.log

#         dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
#         checkpoint=$(basename "$model")            # "checkpoint-112"
#         model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

#         echo "Model name: $model_name"

#         json_path="results/phase1_${model_name}_test_outputs.json"
#         echo "Scoring: $json_path"
#         python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
#     done
# done


###

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


# nohup python phase3_grpo_5.py \
# --model "skt/A.X-4.0-Light" --epochs 8 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --num_iterations 2 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_prompt2_v2" --data_path "data/preprocessed/grpo_train_excluded_서술형_skt_curriculum.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_5.py \
--model "skt/A.X-4.0-Light" --epochs 8 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --num_iterations 2 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_prompt2_v2" --data_path "data/preprocessed/grpo_train_excluded_서술형_skt_curriculum_v2.csv" --system_prompt "$system_prompt"

paths=(
    # "/workspace/korean_culture_QA_2025/models/grpo_v5_A.X-4.0-Light_curri_선다형_단답형_v1_prompt2_v2"
    "/workspace/korean_culture_QA_2025/models/grpo_v5_A.X-4.0-Light_curri_선다형_단답형_v2_prompt2_v2"
)
answer_tag="정답:"

for path in "${paths[@]}"; do
    echo "🔍 상위 경로: $path"
    find "$path" -mindepth 1 -type d | while read -r model; do
        echo "🔍 현재 경로: $model"
        nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
        --system_prompt "$system_prompt" --user_prompt "$user_prompt" --answer_tag "$answer_tag" >> logs/test.log

        dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
        checkpoint=$(basename "$model")            # "checkpoint-112"
        model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

        echo "Model name: $model_name"

        json_path="results/phase1_${model_name}_test_outputs.json"
        echo "Scoring: $json_path"
        python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
    done
done


# nohup python phase3_grpo_5.py \
# --model "K-intelligence/Midm-2.0-Base-Instruct" --epochs 8 --temperature 1.0 --lora_rank 8 --lora_alpha 8 --num_iterations 2 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_prompt2_v2" --data_path "data/preprocessed/grpo_train_excluded_서술형_midm_curriculum.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_5.py \
--model "K-intelligence/Midm-2.0-Base-Instruct" --epochs 8 --temperature 1.0 --lora_rank 8 --lora_alpha 8 --num_iterations 2 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_prompt2_v2" --data_path "data/preprocessed/grpo_train_excluded_서술형_midm_curriculum_v2.csv" --system_prompt "$system_prompt"

paths=(
    # "/workspace/korean_culture_QA_2025/models/grpo_v5_Midm-2.0-Base-Instruct_curri_선다형_단답형_v1_prompt2_v2"
    "/workspace/korean_culture_QA_2025/models/grpo_v5_Midm-2.0-Base-Instruct_curri_선다형_단답형_v2_prompt2_v2"
)
answer_tag="정답:"

for path in "${paths[@]}"; do
    echo "🔍 상위 경로: $path"
    find "$path" -mindepth 1 -type d | while read -r model; do
        echo "🔍 현재 경로: $model"
        nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
        --system_prompt "$system_prompt" --user_prompt "$user_prompt" --answer_tag "$answer_tag" >> logs/test.log

        dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
        checkpoint=$(basename "$model")            # "checkpoint-112"
        model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

        echo "Model name: $model_name"

        json_path="results/phase1_${model_name}_test_outputs.json"
        echo "Scoring: $json_path"
        python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
    done
done
