# !pip install unsloth unsloth_zoo
from unsloth import FastLanguageModel
import torch
import argparse
from rouge_metric import Rouge
from bert_score import score as bert_score
import json
from dotenv import load_dotenv
import os
import wandb
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
from datasets import Dataset
import re
import numpy as np
from vllm import SamplingParams
from reward_func import *

def main():
    parser = argparse.ArgumentParser(description="Phase 3: GRPO Experiment")
    parser.add_argument("--model", default="kakaocorp/kanana-1.5-8b-instruct-2505", help="Model name to use")
    parser.add_argument("--prompt_template", required=True, help="Prompt template to use")
    parser.add_argument("--temperature", default=1.0, type=float, help="Sampling temperature")
    parser.add_argument("--lora_rank", default=8, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=8, type=int, help="LoRA alpha")
    parser.add_argument("--epochs", default=5, type=int, help="epochs")
    parser.add_argument("--system_prompt", default='', type=str, help="system_prompt")
    parser.add_argument("--solution_start", default='', type=str, help="answer start tag")
    parser.add_argument("--data_path", default='data/preprocessed/grpo_train.csv', type=str, help="data_path")
    parser.add_argument("--save_name", default='', type=str, help="save_name")

    args = parser.parse_args()

    # match_format = re.compile(
    #     # rf"{args.solution_start}(.+?)$", re.DOTALL
    #     rf"{args.solution_start}(.+)"       # 태그 라인만 체크
    #     )

    max_seq_length = 3096 # Can increase for longer reasoning traces
    lora_rank = args.lora_rank # Larger rank = smarter, but slower
    lora_alpha = args.lora_alpha # Larger rank = smarter, but slower

    model_name = args.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # Use Qwen3-4B-Base for 4B model
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        load_in_8bit = False,
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_alpha,
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = 3407,
    )


    training_df = pd.read_csv(args.data_path)
    training_df['answer'] = training_df['answer'].astype(str).str.strip()
    training_df['question'] = training_df['question'].astype(str).str.strip()

    prompt_template = args.prompt_template

    prompts = []

    for i in range(len(training_df)):
        row = training_df.iloc[i]
        prompt = prompt_template.format(
            category=row["category"],
            domain=row["domain"],
            topic_keyword=row["topic_keyword"],
            question_type=row["question_type"],
            question=row["question"]
        )
        prompts.append(prompt)

    training_df["prompt"] = prompts

    # 2. Dataset으로 변환
    dataset = Dataset.from_pandas(training_df[["prompt", "answer"]])

    # 3. Chat format으로 변환
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": x["prompt"]}
        ],
        "answer": x["answer"]
    })

    # 확인
    print("Dataset loaded and formatted.")
    print(dataset[0])

    # match_format = re.compile(rf".{5,}{args.solution_start}(.*?)$", re.DOTALL)

    tokenized = dataset.map(
        lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
        batched = True,
    )
    print(tokenizer.decode(tokenized[0]["tokens"]))
    tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

    maximum_length = int(np.quantile(tokenized["L"], 1.0))
    print("Max Length = ", maximum_length)

    # Filter only samples smaller than 90% max length
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    del tokenized
    max_prompt_length = maximum_length + 1 # + 1 just in case!
    max_completion_length = max_seq_length - max_prompt_length

    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 0.95,
        top_k = -1,
        seed = 3407,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )

    num_train_epochs = args.epochs
    save_name = f"grpo_v6_{model_name.split('/')[-1]}_{args.save_name}"

    # ✅ wandb 초기화
    # .env 파일 로드
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    wandb.login(key=wandb_api_key)

    wandb.init(
        project="moducorpus_korea_culture",
        name=save_name,  # W&B에 기록됨
    )

    training_args = GRPOConfig(
        vllm_sampling_params = vllm_sampling_params,
        temperature = args.temperature,
        learning_rate = 1e-5,
        weight_decay = 0.01,
        warmup_ratio = 0.05,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        repetition_penalty = 1.05,
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 1,
        num_generations = 16, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        num_train_epochs = num_train_epochs, # Set to 1 for a full training run
        save_steps = 0.49 / num_train_epochs,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = f"models/{save_name}",
        log_completions = True,
        mask_truncated_completions = True,
        shuffle_dataset = True,
        # For optional training + evaluation
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )
    print(f"dataset:\n{dataset}")
    print(dataset[0]['prompt'][0]['content'])
    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            check_answer,
            penalize_english_overuse
        ],
        args = training_args,
        train_dataset = dataset,

        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        # eval_dataset = new_dataset["test"],
    )
    trainer.train()

if __name__ == "__main__":
    main() 