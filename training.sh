# cd korean_culture_QA_2025

# nohup python phase3_grpo.py \
#     --model "kakaocorp/kanana-1.5-8b-instruct-2505" --temperature 0.8 > phase3_grpo_kanana.log 2>&1 &


nohup python phase3_grpo_v2.py \
    --model "unsloth/Qwen3-8B" --temperature 1.0 --lora_rank 8 > phase3_grpo_qwen.log 2>&1 &