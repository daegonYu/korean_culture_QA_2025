from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# LoRA adapter를 로드 (HuggingFace 혹은 로컬에 위치)
id_ = 140
peft_model = AutoPeftModelForCausalLM.from_pretrained(f"/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형_only_answer/checkpoint-{id_}")
tokenizer = AutoTokenizer.from_pretrained(f"/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형_only_answer/checkpoint-{id_}")

# LoRA를 base 모델에 병합
merged_model = peft_model.merge_and_unload()

# tokenizer도 base 모델에서 불러와 저장
merged_model.save_pretrained(f"/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형_only_answer/checkpoint-{id_}_merged")
tokenizer.save_pretrained(f"/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_선다형_only_answer/checkpoint-{id_}_merged")
