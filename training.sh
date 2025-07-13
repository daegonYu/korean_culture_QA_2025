

# system_prompt="한국의 문화에 기반하여 질문에 정확한 답변을 하십시오.
# 주어진 정보를 참고하여 문제에 가장 적합한 정답을 작성하십시오.

# - 답변 형식
# 답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오."


# nohup python phase3_grpo_3.py \
# --model "skt/A.X-4.0-Light" --epochs 5 --temperature 1.0 --lora_rank 16 --solution_start "정답:" --save_name "근거_선다형_단답형" --data_path "data/preprocessed/grpo_train_excluded_서술형.csv" --system_prompt "$system_prompt" > logs/phase3_grpo_skt_2.log 2>&1



# system_prompt="주어진 문제에 적절한 답변을 서술하십시오."

# nohup python phase3_grpo_3.py \
# --model "skt/A.X-4.0-Light" --epochs 5 --temperature 1.0 --lora_rank 8 --solution_start "" --save_name "서술형2" --data_path "data/preprocessed/grpo_train_서술형.csv" --system_prompt "$system_prompt" > logs/phase3_grpo_skt_3.log 2>&1

# system_prompt="주어진 문제에 적절한 답변을 서술하십시오.

# - 답변 형식
# 답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오."


# nohup python phase3_grpo_3.py \
# --model "skt/A.X-4.0-Light" --epochs 6 --temperature 1.2 --lora_rank 16 --solution_start "정답:" --save_name "근거_ALL_2" --data_path "data/preprocessed/grpo_train.csv" --system_prompt "$system_prompt"

system_prompt="주어진 문제에 적절한 답변을 서술하십시오.

- 답변 형식
답변 근거를 '답변 근거:'에 서술한 뒤 최종 정답을 '정답:'에 작성하십시오."


nohup python phase3_grpo_3.py \
--model "skt/A.X-4.0-Light" --epochs 12 --temperature 1.2 --lora_rank 16 --solution_start "정답:" --save_name "근거_ALL_2_epochs-12" --data_path "data/preprocessed/grpo_train.csv" --system_prompt "$system_prompt"