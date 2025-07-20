#!/usr/bin/env bash


# Phase 2: Scoring
json_path="results/test/prompt_v2/phase1_kanana-1.5-8b-instruct-2505_intermediate_results_394_test_format_converted.json"
echo "Scoring: $json_path"
answer_tag=""
python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"

#!/usr/bin/env bash
set -e

model_list=(
#    "skt/A.X-4.0-Light"
#    "kakaocorp/kanana-1.5-8b-instruct-2505"
#    "Qwen/Qwen3-8B"
    "/workspace/korean_culture_QA_2025/models/grpo_v2_A.X-4.0-Light_선다형_단답형/checkpoint-28"
    "/workspace/korean_culture_QA_2025/models/grpo_v2_A.X-4.0-Light_선다형_단답형/checkpoint-56"
    "/workspace/korean_culture_QA_2025/models/grpo_v2_A.X-4.0-Light_선다형_단답형/checkpoint-112"
    "/workspace/korean_culture_QA_2025/models/grpo_v2_A.X-4.0-Light_선다형_단답형/checkpoint-168"
    "/workspace/korean_culture_QA_2025/models/grpo_v2_A.X-4.0-Light_선다형_단답형/checkpoint-228"
)

for model in "${model_list[@]}"
do
    echo "Running model: $model"
    system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.

사용자가 입력한 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.
- 질문 유형(question_type): '선다형', '단답형'
선다형 문제의 경우, 가장 정답과 가까운 번호를 선택하십시오.
단답형 문제의 경우, 단어 (구)로 작성하십시오.

- 답변 형식
당신은 사용자의 질문에 대해 먼저 머릿속으로 사고 과정을 거친 뒤, 그 과정을 설명하고 최종 답변을 제공합니다.  
사고 과정은 <think>...</think> 태그 안에, 최종적인 답변은 <answer>...</answer> 태그 안에 작성하세요."

    user_prompt="주어진 질문에 적절한 답변을 해주세요.\n\ncategory: {category}\ndomain: {domain}\ntopic_keyword: {topic_keyword}\nquestion_type: {question_type}\n\n질문: {question}\n\n답변:"
    answer_tag="<answer>"

    dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
    checkpoint=$(basename "$model")            # "checkpoint-112"
    model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

    # Phase 2: Scoring
    json_path="results/phase1_${model_name}_test_outputs.json"
    echo "Scoring: $json_path"
    python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
done



model_list=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_ALL/checkpoint-37"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_ALL/checkpoint-74"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_ALL/checkpoint-148"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_ALL/checkpoint-185"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_ALL/checkpoint-259"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_ALL/checkpoint-333"
)

for model in "${model_list[@]}"
do
    echo "Running model: $model"
    system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오. 주어진 문제에 대해 적절한 답변을 '정답:' 뒤에 작성하십시오."
    user_prompt="주어진 질문에 적절한 답변을 해주세요.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n질문 유형: {question_type}\n\n질문: {question}\n\n답변:"
    answer_tag="정답:"

    dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
    checkpoint=$(basename "$model")            # "checkpoint-112"
    model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

    # Phase 2: Scoring
    json_path="results/phase1_${model_name}_test_outputs.json"
    echo "Scoring: $json_path"
    python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
done


model_list=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL/checkpoint-37"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL/checkpoint-74"
)

for model in "${model_list[@]}"
do
    echo "Running model: $model"
    system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오. 질문 유형이 서술형인 경우 답변 근거 없이 정답을 작성하십시오."
    user_prompt="주어진 질문에 적절한 답변을 해주세요.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n질문 유형: {question_type}\n\n질문: {question}\n\n답변:"
    answer_tag="정답:"

    dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
    checkpoint=$(basename "$model")            # "checkpoint-112"
    model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

    # Phase 2: Scoring
    json_path="results/phase1_${model_name}_test_outputs.json"
    echo "Scoring: $json_path"
    python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
done


model_list=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL_2/checkpoint-37"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL_2/checkpoint-74"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL_2/checkpoint-111"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL_2/checkpoint-148"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL_2/checkpoint-185"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL_2/checkpoint-259"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL_2/checkpoint-333"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL_2/checkpoint-407"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL_2_epochs-12/checkpoint-444"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_ALL_2_epochs-12/checkpoint-481"
)

for model in "${model_list[@]}"
do
    echo "Running model: $model"
    system_prompt="주어진 문제에 적절한 답변을 서술하십시오.

- 답변 형식
답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오."
    user_prompt="주어진 질문에 적절한 답변을 해주세요.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n질문 유형: {question_type}\n\n질문: {question}\n\n답변:"
    answer_tag="정답:"

    dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
    checkpoint=$(basename "$model")            # "checkpoint-112"
    model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

    # Phase 2: Scoring
    json_path="results/phase1_${model_name}_test_outputs.json"
    echo "Scoring: $json_path"
    python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
done





model_list=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_단답형/checkpoint-10"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_단답형/checkpoint-20"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_단답형/checkpoint-30"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_단답형/checkpoint-40"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_단답형/checkpoint-50"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_단답형/checkpoint-70"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_단답형/checkpoint-90"
)

for model in "${model_list[@]}"
do
    echo "Running model: $model"
    system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오."
    user_prompt="주어진 질문에 적절한 답변을 해주세요.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n질문 유형: {question_type}\n\n질문: {question}\n\n답변:"
    answer_tag="정답:"

    dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
    checkpoint=$(basename "$model")            # "checkpoint-112"
    model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

    # Phase 2: Scoring
    json_path="results/phase1_${model_name}_test_outputs.json"
    echo "Scoring: $json_path"
    python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
done



model_list=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_선다형_단답형/checkpoint-29"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_선다형_단답형/checkpoint-58"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_선다형_단답형/checkpoint-87"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_선다형_단답형/checkpoint-116"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_선다형_단답형/checkpoint-145"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_선다형_단답형/checkpoint-174"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_선다형_단답형/checkpoint-232"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_근거_선다형_단답형/checkpoint-285"
)

for model in "${model_list[@]}"
do
    echo "Running model: $model"
    system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오."
    user_prompt="주어진 질문에 적절한 답변을 해주세요.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n질문 유형: {question_type}\n\n질문: {question}\n\n답변:"
    answer_tag="정답:"

    dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
    checkpoint=$(basename "$model")            # "checkpoint-112"
    model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

    # Phase 2: Scoring
    json_path="results/phase1_${model_name}_test_outputs.json"
    echo "Scoring: $json_path"
    python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
done




model_list=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_단답형_epoch_8/checkpoint-20"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_단답형_epoch_8/checkpoint-40"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_단답형_epoch_8/checkpoint-60"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_단답형_epoch_8/checkpoint-80"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_단답형_epoch_8/checkpoint-100"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_단답형_epoch_8/checkpoint-120"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_단답형_epoch_8/checkpoint-140"
)

for model in "${model_list[@]}"
do
    echo "Running model: $model"
    system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
<think></think> 태그 안에 문제를 풀기 위한 논리적 사고 후 최종 답변은 <answer></answer> 태그 안에 작성하세요."
    user_prompt="주어진 질문에 적절한 답변을 해주세요.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n질문 유형: {question_type}\n\n질문: {question}\n\n답변:"
    answer_tag="<answer>"

    dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
    checkpoint=$(basename "$model")            # "checkpoint-112"
    model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

    # Phase 2: Scoring
    json_path="results/phase1_${model_name}_test_outputs.json"
    echo "Scoring: $json_path"
    python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
done




model_list=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_서술형/checkpoint-18"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_서술형/checkpoint-27"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_서술형/checkpoint-36"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_서술형/checkpoint-45"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_서술형/checkpoint-54"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_서술형/checkpoint-72"
)

for model in "${model_list[@]}"
do
    echo "Running model: $model"
    system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 문제에 대해 적절한 답변을 서술하십시오."
    user_prompt="주어진 질문에 적절한 답변을 해주세요.\n\ncategory: {category}\ndomain: {domain}\ntopic_keyword: {topic_keyword}\nquestion_type: {question_type}\n\n질문: {question}\n\n답변:"
    answer_tag=""

    dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
    checkpoint=$(basename "$model")            # "checkpoint-112"
    model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

    # Phase 2: Scoring
    json_path="results/phase1_${model_name}_test_outputs.json"
    echo "Scoring: $json_path"
    python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
done



model_list=(
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형/checkpoint-40"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형/checkpoint-60"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형/checkpoint-80"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형/checkpoint-100"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형/checkpoint-140"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형/checkpoint-180"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형_temperature-07/checkpoint-100"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형_temperature-07/checkpoint-140"
    "/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형_temperature-07/checkpoint-180"
)

for model in "${model_list[@]}"
do
    echo "Running model: $model"
    system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

- 답변 형식
<think></think> 태그 안에 문제를 풀기 위한 논리적 사고 후 최종 답변은 <answer></answer> 태그 안에 선다형 번호만 작성하세요."
    user_prompt="주어진 질문에 적절한 답변을 해주세요.\n\ncategory: {category}\ndomain: {domain}\ntopic_keyword: {topic_keyword}\nquestion_type: {question_type}\n\n질문: {question}\n\n답변:"
    answer_tag="<answer>"


    dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
    checkpoint=$(basename "$model")            # "checkpoint-112"
    model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

    echo "Model name: $model_name"

    # Phase 2: Scoring
    json_path="results/phase1_${model_name}_test_outputs.json"
    echo "Scoring: $json_path"
    python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
done


model_list=(
    "skt/A.X-4.0-Light"
    "skt/A.X-3.1-Light"
    "kakaocorp/kanana-1.5-8b-instruct-2505"
    "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    # "Qwen/Qwen3-8B"
)


for model in "${model_list[@]}"
do
    echo "Running model: $model"
    system_prompt="주어진 문제에 적절한 답변을 서술하십시오.

- 답변 형식
답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오."
    user_prompt="주어진 질문에 적절한 답변을 해주세요.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n질문 유형: {question_type}\n\n질문: {question}\n\n답변:"
    answer_tag="정답:"


    dir_name=$(basename $(dirname "$model"))   # "grpo_v2_A.X-4.0-Light_선다형_단답형"
    checkpoint=$(basename "$model")            # "checkpoint-112"
    model_name="${dir_name}_${checkpoint}"     # "grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-112"

    echo "Model name: $model_name"

    # Phase 2: Scoring
    json_path="results/phase1_${model_name}_test_outputs.json"
    echo "Scoring: $json_path"
    python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"
done





