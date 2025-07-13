cd korean_culture_QA_2025




nohup python phase3_grpo.py \
    --model "kakaocorp/kanana-1.5-8b-instruct-2505" --temperature 0.8 > phase3_grpo_kanana.log 2>&1


nohup python phase3_grpo_v2.py \
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