#!/usr/bin/env bash
set -e

cd korean_culture_QA_2025


nohup python -m scripts.phase3_grpo \
    --model "kakaocorp/kanana-1.5-8b-instruct-2505" --temperature 0.8 > phase3_grpo_kanana.log 2>&1


nohup python -m phase3_grpo_v2 \
    --model "unsloth/Qwen3-8B" --temperature 1.0 --lora_rank 8 > phase3_grpo_qwen.log 2>&1





# 선다형
system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
<think></think> 태그 안에 문제를 풀기 위한 논리적 사고 후 최종 답변은 <answer></answer> 태그 안에 선다형 번호만 작성하세요."


nohup python phase3_grpo_.py \
--model "skt/A.X-4.0-Light" --epochs 5 --temperature 0.8 --lora_rank 8 --solution_start "<answer>" --save_name "선다형" --data_path "data/preprocessed/grpo_train_선다형.csv" --system_prompt "$system_prompt" > logs/phase3_grpo_skt_1.log 2>&1



# 단답형
system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
<think></think> 태그 안에 문제를 풀기 위한 논리적 사고 후 최종 답변은 <answer></answer> 태그 안에 작성하세요."


nohup python phase3_grpo_.py \
--model "skt/A.X-4.0-Light" --epochs 5 --temperature 0.8 --lora_rank 8 --solution_start "<answer>" --save_name "단답형" --data_path "data/preprocessed/grpo_train_단답형.csv" --system_prompt "$system_prompt" > logs/phase3_grpo_skt_2.log 2>&1


system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
<think></think> 태그 안에 문제를 풀기 위한 논리적 사고 후 최종 답변은 <answer></answer> 태그 안에 작성하세요."


nohup python phase3_grpo_2.py \
--model "skt/A.X-4.0-Light" --epochs 8 --temperature 0.8 --lora_rank 8 --solution_start "<answer>" --save_name "단답형_epoch_8" --data_path "data/preprocessed/grpo_train_단답형.csv" --system_prompt "$system_prompt" > logs/phase3_grpo_skt_2.log 2>&1


system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오."


nohup python phase3_grpo_2.py \
--model "skt/A.X-4.0-Light" --epochs 5 --temperature 1.0 --lora_rank 16 --solution_start "정답:" --save_name "근거_단답형" --data_path "data/preprocessed/grpo_train_단답형.csv" --system_prompt "$system_prompt" > logs/phase3_grpo_skt_2.log 2>&1


# 선다형, 단답형
nohup python phase3_grpo_v2.py \
--model "skt/A.X-4.0-Light" --temperature 0.7 --lora_rank 8 --system_prompt "한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.

사용자가 입력한 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.
- 질문 유형(question_type): '선다형', '단답형'
선다형 문제의 경우, 가장 정답과 가까운 번호를 선택하십시오.
단답형 문제의 경우, 단어 (구)로 작성하십시오.

- 답변 형식
당신은 사용자의 질문에 대해 먼저 머릿속으로 사고 과정을 거친 뒤, 그 과정을 설명하고 최종 답변을 제공합니다.  
사고 과정은 `<think>...</think>` 태그 안에, 최종적인 답변은 `<answer>...</answer>` 태그 안에 작성하세요"> phase3_grpo_skt.log 2>&1


system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오."


nohup python phase3_grpo_3.py \
--model "skt/A.X-4.0-Light" --epochs 5 --temperature 1.0 --lora_rank 16 --solution_start "정답:" --save_name "근거_선다형_단답형" --data_path "data/preprocessed/grpo_train_excluded_서술형.csv" --system_prompt "$system_prompt" > logs/phase3_grpo_skt_2.log 2>&1




# 서술형
system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 문제에 대해 적절한 답변을 서술하십시오."

nohup python phase3_grpo_.py \
--model "skt/A.X-4.0-Light" --epochs 5 --temperature 0.8 --lora_rank 8 --solution_start "" --save_name "서술형" --data_path "data/preprocessed/grpo_train_서술형.csv" --system_prompt "$system_prompt" > logs/phase3_grpo_skt_3.log 2>&1



system_prompt="주어진 문제에 적절한 답변을 서술하십시오."

nohup python phase3_grpo_3.py \
--model "skt/A.X-4.0-Light" --epochs 5 --temperature 1.0 --lora_rank 8 --solution_start "" --save_name "서술형" --data_path "data/preprocessed/grpo_train_서술형.csv" --system_prompt "$system_prompt" > logs/phase3_grpo_skt_3.log 2>&1



# ALL
system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 문제에 대해 적절한 답변을 '정답:' 뒤에 작성하십시오."

nohup python phase3_grpo_2.py \
--model "skt/A.X-4.0-Light" --epochs 5 --temperature 0.8 --lora_rank 8 --solution_start "정답:" --save_name "ALL" --data_path "data/preprocessed/grpo_train.csv" --system_prompt "$system_prompt" > logs/phase3_grpo_skt_4.log 2>&1



system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오. 질문 유형이 서술형인 경우 답변 근거 없이 정답을 작성하십시오."


nohup python phase3_grpo_3.py \
--model "skt/A.X-4.0-Light" --epochs 6 --temperature 1.0 --lora_rank 16 --solution_start "정답:" --save_name "근거_ALL" --data_path "data/preprocessed/grpo_train.csv" --system_prompt "$system_prompt"


system_prompt="주어진 문제에 적절한 답변을 서술하십시오.

- 답변 형식
답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오."


nohup python phase3_grpo_3.py \
--model "skt/A.X-4.0-Light" --epochs 6 --temperature 1.2 --lora_rank 16 --solution_start "정답:" --save_name "근거_ALL_2" --data_path "data/preprocessed/grpo_train.csv" --system_prompt "$system_prompt"


system_prompt="당신은 한국의 문화와 관련된 문제를 전문적으로 풀이해주는 문제 해설가입니다.  
사용자가 입력한 문제에 대해 정확하고 친절하게 **문제 해설**과 **정답**을 제시하세요.  
답변 형식은 반드시 다음과 같이 작성하세요:  
문제 해설: ...  
정답: ..."
user_prompt="아래 문제를 해설해주고 정답을 알려줘.\n반드시 문제 해설 다음에 정답을 작성해줘.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n문제 유형: {question_type}\n\n문제: {question}\n\n답변:"


nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 0.7 --lora_rank 8 --lora_alpha 8 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "선다형_v2" --data_path "data/preprocessed/grpo_train_선다형.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 8 --lora_alpha 4 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "선다형_v3" --data_path "data/preprocessed/grpo_train_선다형.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 8 --lora_alpha 8 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "선다형_v4" --data_path "data/preprocessed/grpo_train_선다형.csv" --system_prompt "$system_prompt"


nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 0.7 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "선다형_v5" --data_path "data/preprocessed/grpo_train_선다형.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 16 --lora_alpha 8 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "선다형_v6" --data_path "data/preprocessed/grpo_train_선다형.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "선다형_v7" --data_path "data/preprocessed/grpo_train_선다형.csv" --system_prompt "$system_prompt"


nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 0.7 --lora_rank 32 --lora_alpha 32 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "선다형_v8" --data_path "data/preprocessed/grpo_train_선다형.csv" --system_prompt "$system_prompt"


nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 32 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "선다형_v9" --data_path "data/preprocessed/grpo_train_선다형.csv" --system_prompt "$system_prompt"


nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 32 --lora_alpha 32 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "선다형_v10" --data_path "data/preprocessed/grpo_train_선다형.csv" --system_prompt "$system_prompt"


system_prompt="당신은 한국의 문화와 관련된 문제를 전문적으로 풀이해주는 문제 해설가입니다.  
사용자가 입력한 문제에 대해 정확하고 친절하게 **문제 해설**과 **정답**을 제시하세요.  
답변 형식은 반드시 다음과 같이 작성하세요:  
문제 해설: ...  
정답: ...

※ 정답 작성 형식 안내  
- 선다형 문제일 경우: **정답 번호**를 작성하세요.  
- 단답형 문제일 경우:  
    - 정답이 1개인 경우: 괄호, 한자 없이 **한글** 또는 **영어**를 사용하세요. (예: 사과 또는 apple)  
    - 정답이 여러 개인 경우: 쉼표(,)로 구분해 나열하세요. (예: 사과, 배)  
    - 순서 배열 문제인 경우: '-'로 구분해 정확한 순서를 유지해 나열하세요. (예: ㄱ-ㄴ-ㄷ-ㄹ)"

user_prompt="아래 문제를 해설해주고 정답을 알려줘.\n반드시 문제 해설 다음에 정답을 작성해줘.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n문제 유형: {question_type}\n\n문제: {question}\n\n답변:"

nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum_v2.csv" --system_prompt "$system_prompt"

paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v1"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v2"
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

###

system_prompt="당신은 한국의 문화와 관련된 문제를 전문적으로 풀이해주는 문제 해설가입니다.  
사용자가 입력한 문제에 대해 정확하고 친절하게 **문제 해설**과 **정답**을 제시하세요.  
답변 형식은 반드시 다음과 같이 작성하세요:  
문제 해설: ...  
정답: ...

※ 정답 작성 형식 안내  
- 선다형 문제일 경우: **정답 번호**를 작성하세요.  
- 단답형 문제일 경우:  
    - 정답이 1개인 경우: 괄호, 한자 없이 **한글** 또는 **영어**를 사용하세요. (예: 사과 또는 apple)  
    - 정답이 여러 개인 경우: 쉼표(,)로 구분해 나열하세요. (예: 사과, 배)  
    - 순서 배열 문제인 경우: '-'로 구분해 정확한 순서를 유지해 나열하세요. (예: ㄱ-ㄴ-ㄷ-ㄹ)"

user_prompt="아래 문제를 해설해주고 정답을 알려줘.\n반드시 문제 해설 다음에 정답을 작성해줘.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n문제 유형: {question_type}\n\n문제: {question}\n\n답변:"

nohup python phase3_grpo_4.py \
--model "kakaocorp/kanana-1.5-8b-instruct-2505" --epochs 5 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_4.py \
--model "kakaocorp/kanana-1.5-8b-instruct-2505" --epochs 5 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum_v2.csv" --system_prompt "$system_prompt"

paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_kanana-1.5-8b-instruct-2505_curri_선다형_단답형_v1"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_kanana-1.5-8b-instruct-2505_curri_선다형_단답형_v2"
)
answer_tag="정답:"

### 

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

nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 5 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_lr" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 5 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_lr" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum_v2.csv" --system_prompt "$system_prompt"

paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_X-4.0-Light_curri_선다형_단답형_v1_lr"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_X-4.0-Light_curri_선다형_단답형_v2_lr"
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


nohup python phase3_grpo_5.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_5.py \
--model "skt/A.X-4.0-Light" --epochs 4 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_curriculum_v2.csv" --system_prompt "$system_prompt"

paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v5_A.X-4.0-Light_curri_선다형_단답형_v1_prompt2"
    "/workspace/korean_culture_QA_2025/models/grpo_v5_A.X-4.0-Light_curri_선다형_단답형_v2_prompt2"
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


nohup python phase3_grpo_5.py \
--model "K-intelligence/Midm-2.0-Base-Instruct" --epochs 4 --temperature 1.0 --lora_rank 8 --lora_alpha 8 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_midm_curriculum.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_5.py \
--model "K-intelligence/Midm-2.0-Base-Instruct" --epochs 4 --temperature 1.0 --lora_rank 8 --lora_alpha 8 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_midm_curriculum_v2.csv" --system_prompt "$system_prompt"

paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v5_Midm-2.0-Base-Instruct_curri_선다형_단답형_v1_prompt2"
    "/workspace/korean_culture_QA_2025/models/grpo_v5_Midm-2.0-Base-Instruct_curri_선다형_단답형_v2_prompt2"
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

answer_tag="정답:"

# nohup python phase3_grpo_5.py \
# --model "skt/A.X-4.0-Light" --epochs 8 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --num_iterations 2 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_prompt2_v2" --data_path "data/preprocessed/grpo_train_excluded_서술형_skt_curriculum.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_5.py \
--model "skt/A.X-4.0-Light" --epochs 8 --temperature 1.2 --lora_rank 16 --lora_alpha 16 --num_iterations 2 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_prompt2_v2" --data_path "data/preprocessed/grpo_train_excluded_서술형_skt_curriculum_v2.csv" --system_prompt "$system_prompt"

paths=(
    # "/workspace/korean_culture_QA_2025/models/grpo_v5_A.X-4.0-Light_curri_선다형_단답형_v1_prompt2_v2"
    "/workspace/korean_culture_QA_2025/models/grpo_v5_A.X-4.0-Light_curri_선다형_단답형_v2_prompt2_v2"
)

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

answer_tag="정답:"


nohup python phase3_grpo_4.py \
--model "skt/A.X-4.0-Light" --epochs 8 --temperature 1.0 --lora_rank 32 --lora_alpha 32 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_skt_curriculum_v2.csv" --system_prompt "$system_prompt"

paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v4_A.X-4.0-Light_curri_선다형_단답형_v2_prompt2"
)

for path in "${paths[@]}"; do
    echo "🔍 상위 경로: $path"
    find "$path" -mindepth 1 -type d | while read -r model; do
        echo "🔍 현재 경로: $model"
        nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
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



nohup python phase3_grpo_4.py \
--model "K-intelligence/Midm-2.0-Base-Instruct" --epochs 8 --temperature 1.0 --lora_rank 32 --lora_alpha 32 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_midm_curriculum.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_4.py \
--model "K-intelligence/Midm-2.0-Base-Instruct" --epochs 8 --temperature 1.0 --lora_rank 32 --lora_alpha 32 --solution_start "정답:" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v2_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_midm_curriculum_v2.csv" --system_prompt "$system_prompt"


paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v4_Midm-2.0-Base-Instruct_curri_선다형_단답형_v1_prompt2"
    "/workspace/korean_culture_QA_2025/models/grpo_v4_Midm-2.0-Base-Instruct_curri_선다형_단답형_v2_prompt2"
)

for path in "${paths[@]}"; do
    echo "🔍 상위 경로: $path"
    find "$path" -mindepth 1 -type d | while read -r model; do
        echo "🔍 현재 경로: $model"
        nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
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

answer_tag="정답:"

nohup accelerate launch --config_file accelerate/fsdp_config.yaml phase3_grpo_4_fsdp.py \
  --model "K-intelligence/Midm-2.0-Base-Instruct" \
  --temperature 1.0 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --epochs 8 \
  --system_prompt "$system_prompt" \
  --prompt_template "$user_prompt" \
  --solution_start "$answer_tag" \
  --data_path "data/preprocessed/grpo_train_excluded_서술형_midm_curriculum.csv" \
  --save_name "curri_선다형_단답형_v2_prompt2_fsdp"

###

nohup python phase3_grpo_6.py \
--model "K-intelligence/Midm-2.0-Base-Instruct" --epochs 12 --temperature 1.0 --lora_rank 32 --lora_alpha 32 --solution_start "$answer_tag" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_midm_curriculum.csv" --system_prompt "$system_prompt"

nohup python phase3_grpo_6.py \
--model "skt/A.X-4.0-Light" --epochs 12 --temperature 1.0 --lora_rank 64 --lora_alpha 64 --solution_start "$answer_tag" --prompt_template "$user_prompt" --save_name "curri_선다형_단답형_v1_prompt2" --data_path "data/preprocessed/grpo_train_excluded_서술형_skt_curriculum.csv" --system_prompt "$system_prompt"


paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v6_Midm-2.0-Base-Instruct_curri_선다형_단답형_v1_prompt2"
    "/workspace/korean_culture_QA_2025/models/grpo_v6_A.X-4.0-Light_curri_선다형_단답형_v1_prompt2"
)

for path in "${paths[@]}"; do
    echo "🔍 상위 경로: $path"
    find "$path" -mindepth 1 -type d | while read -r model; do
        echo "🔍 현재 경로: $model"
        nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
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


paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v4_A.X-4.0-Light_curri_선다형_단답형_v1_prompt2"
    "/workspace/korean_culture_QA_2025/models/grpo_v4_Midm-2.0-Base-Instruct_curri_선다형_단답형_v1_prompt2"
)

for path in "${paths[@]}"; do
    echo "🔍 상위 경로: $path"
    find "$path" -mindepth 1 -type d | while read -r model; do
        echo "🔍 현재 경로: $model"
        nohup python run_phase1.py --model "$model" --use_test --use_wandb --use_lora \
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

answer_tag="정답:"


nohup accelerate launch --config_file accelerate/fsdp_config.yaml phase3_grpo_6_fsdp.py \
  --model "K-intelligence/Midm-2.0-Base-Instruct" \
  --temperature 1.1 \
  --epochs 15 \
  --system_prompt "$system_prompt" \
  --prompt_template "$user_prompt" \
  --solution_start "$answer_tag" \
  --data_path "/workspace/korean_culture_QA_2025/data/preprocessed/grpo_train_excluded_서술형_midm_curriculum.csv" \
  --save_name "curri_선다형_단답형_v1_prompt2_FFT" \
  --vllm_gpu_memory_utilization 0.6


paths=(
    "/workspace/korean_culture_QA_2025/models/grpo_v6_Midm-2.0-Base-Instruct_curri_선다형_단답형_v1_prompt2_FFT"
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

user_prompt="한국의 문화와 관련된 아래 문제를 단계별로 자세히 해설해주고, 마지막에 정답을 작성해줘.  

키워드: {topic_keyword}  
문제 유형: {question_type}  
문제: {question}"

answer_tag="정답:"


nohup python -m scripts.phase3_grpo_6 \
  --model "trillionlabs/Tri-7B" \
  --temperature 1.0 \
  --epochs 10 \
  --lora_rank 128 \
  --lora_alpha 128 \
  --loss_type 'bnpo' \
  --system_prompt "$system_prompt" \
  --prompt_template "$user_prompt" \
  --solution_start "$answer_tag" \
  --train_data "/workspace/korean_culture_QA_2025/data/preprocessed/grpo_train_excluded_서술형_trillion_curriculum_v1.csv" \
  --valid_data "/workspace/korean_culture_QA_2025/data/preprocessed/original_dev_excluded_서술형.csv" \
  --do_eval \
  --save_name "original_train_선다형_단답형_v1_prompt2_bnpo"

nohup python -m scripts.phase3_grpo_6 \
  --model "trillionlabs/Tri-7B" \
  --temperature 1.0 \
  --epochs 10 \
  --lora_rank 128 \
  --lora_alpha 128 \
  --loss_type 'dr_grpo' \
  --system_prompt "$system_prompt" \
  --prompt_template "$user_prompt" \
  --solution_start "$answer_tag" \
  --train_data "/workspace/korean_culture_QA_2025/data/preprocessed/grpo_train_excluded_서술형_trillion_curriculum_v1.csv" \
  --valid_data "/workspace/korean_culture_QA_2025/data/preprocessed/original_dev_excluded_서술형.csv" \
  --do_eval \
  --save_name "original_train_선다형_단답형_v1_prompt2_dr_grpo"

nohup python -m scripts.phase3_grpo_6 \
  --model "trillionlabs/Tri-7B" \
  --temperature 1.0 \
  --epochs 10 \
  --lora_rank 128 \
  --lora_alpha 128 \
  --loss_type 'gspo' \
  --system_prompt "$system_prompt" \
  --prompt_template "$user_prompt" \
  --solution_start "$answer_tag" \
  --train_data "/workspace/korean_culture_QA_2025/data/preprocessed/grpo_train_excluded_서술형_trillion_curriculum_v1.csv" \
  --valid_data "/workspace/korean_culture_QA_2025/data/preprocessed/original_dev_excluded_서술형.csv" \
  --do_eval \
  --save_name "original_train_선다형_단답형_v1_prompt2_gspo"