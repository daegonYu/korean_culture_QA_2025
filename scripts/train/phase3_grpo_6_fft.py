import argparse
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
import wandb


# import unsloth
from peft import PeftConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from utils.reward_func import match_format_exactly, check_answer, penalize_english_overuse

warnings.filterwarnings('ignore')
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Phase 3: GRPO Experiment")
    parser.add_argument("--model", default="kakaocorp/kanana-1.5-8b-instruct-2505", help="Model name to use")
    parser.add_argument("--prompt_template", required=True, help="Prompt template to use")
    parser.add_argument("--temperature", default=1.0, type=float, help="Sampling temperature")
    parser.add_argument("--epochs", default=8, type=int, help="epochs")
    parser.add_argument("--epsilon", default=0.2, type=float, help="epsilon")
    parser.add_argument("--epsilon_high", default=0.28, type=float, help="epsilon_high")
    parser.add_argument("--system_prompt", default='', type=str, help="system_prompt")
    parser.add_argument("--solution_start", default='', type=str, help="answer start tag")
    parser.add_argument("--loss_type", default='bnpo', type=str, help="setting loss type")
    parser.add_argument("--importance_sampling_level", default='token', type=str, help="importance_sampling_level")
    parser.add_argument("--train_data", default='data/preprocessed/grpo_train.csv', type=str, help="train data path")
    parser.add_argument("--valid_data", default='data/preprocessed/original_dev_excluded_서술형.csv', type=str, help="validation data path")
    parser.add_argument('--do_eval', action='store_true', help="start validation")
    parser.add_argument("--save_name", default='', type=str, help="save_name")

    args = parser.parse_args()

    # if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE","1")) > 1:
    #     local_rank = int(os.environ["LOCAL_RANK"])
    #     torch.cuda.set_device(local_rank)   # ★ init_process_group 전에!
    #     print(f"[rank {os.environ.get('RANK','0')}] local_rank={local_rank}, cuda_current={torch.cuda.current_device()}")

    max_seq_length = 1200 # Can increase for longer reasoning traces
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        # causal LM에서 흔히 pad=eos로 설정
        tokenizer.pad_token = tokenizer.eos_token
    # 길이 긴 프롬프트에서 안전
    tokenizer.padding_side = "left"

    # dtype은 환경에 맞게 자동/선택
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    # 학습 시 반드시 켜주기(메모리 절약)
    model.config.use_cache = False  # gradient checkpointing 시 권장


    training_df = pd.read_csv(args.train_data)
    training_df['answer'] = training_df['answer'].astype(str).str.strip()
    training_df['question'] = training_df['question'].astype(str).str.strip()

    valid_df = pd.read_csv(args.valid_data)
    valid_df['answer'] = valid_df['answer'].astype(str).str.strip()
    valid_df['question'] = valid_df['question'].astype(str).str.strip()
    
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

    prompts = []
    for i in range(len(valid_df)):
        row = valid_df.iloc[i]
        prompt = prompt_template.format(
            category=row["category"],
            domain=row["domain"],
            topic_keyword=row["topic_keyword"],
            question_type=row["question_type"],
            question=row["question"]
        )
        prompts.append(prompt)
    valid_df["prompt"] = prompts


    # 2. Dataset으로 변환
    train_dataset = Dataset.from_pandas(training_df[["prompt", "answer"]])
    valid_dataset = Dataset.from_pandas(valid_df[["prompt", "answer"]])

    # 3. Chat format으로 변환
    train_dataset = train_dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": x["prompt"]}
        ],
        "answer": x["answer"]
    })
    valid_dataset = valid_dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": x["prompt"]}
        ],
        "answer": x["answer"]
    })

    print("train_dataset loaded and formatted.")
    print(train_dataset[0])

    train_tokenized = train_dataset.map(
        lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
        batched = True,
    )
    print(tokenizer.decode(train_tokenized[0]["tokens"]))
    train_tokenized = train_tokenized.map(lambda x: {"L" : len(x["tokens"])})

    maximum_length = int(np.quantile(train_tokenized["L"], 0.97))
    print("Max Length = ", maximum_length)

    train_dataset = train_dataset.select(np.where(np.array(train_tokenized["L"]) <= maximum_length)[0])
    del train_tokenized

    max_prompt_length = maximum_length + 1 # + 1 just in case!
    max_completion_length = max_seq_length - max_prompt_length

    num_train_epochs = args.epochs

    # ✅ wandb 초기화
    # .env 파일 로드
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    wandb.login(key=wandb_api_key)

    wandb.init(
        project="moducorpus_korea_culture",
        name=f"grpo_v6_{args.model.split('/')[-1]}_{args.save_name}",
    )

    # vLLM SamplingParams (GRPO는 내부에서 vllm 호출)
    vllm_sampling_params = SamplingParams(
        temperature=args.temperature,
        min_p=0.1,
        top_p=0.95,
        top_k=-1,
        seed=3407,
        # eos로 자동 종료되므로 stop 문자열은 생략 (특수토큰 문자열 미정일 때 안전)
        include_stop_str_in_output=True,
        repetition_penalty=1.05,
    )

    # 6) GRPOConfig — **GRPOConfig 제공 필드만 사용**
    training_args = GRPOConfig(
        # --- TrainingArguments 계열 ---
        output_dir=f"models/grpo_v6_{args.model.split('/')[-1]}_{args.save_name}",
        report_to= "wandb",
        learning_rate=1e-6,
        weight_decay=0.01,
        warmup_ratio=0.0,
        lr_scheduler_type="cosine",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        save_only_model=True,
        save_strategy="epoch",
        # save_steps=3,
        eval_strategy=("epoch" if args.do_eval else "no"),  # 주의: GRPOConfig는 eval_strategy 사용
        # save_total_limit=1,
        optim="paged_adamw_8bit",
        bf16=bool(torch.cuda.is_available()),
        # gradient_checkpointing=True,      # FSDP와 동시 설정 시 에러
        metric_for_best_model=("eval/reward" if args.do_eval else None),
        greater_is_better=(True if args.do_eval else None),

        # --- GRPO 전용/데이터 전처리 ---
        num_generations=8,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        shuffle_dataset=True,

        # --- 생성 파라미터(모두 GRPOConfig 필드) ---
        temperature=args.temperature,
        top_p=0.95,
        top_k=None,
        min_p=0.1,
        repetition_penalty=1.05,
        # generation_kwargs=None,  # 필요시 추가(충돌 시 여기가 우선)

        # --- vLLM 비활성(필드 자체는 GRPOConfig에 존재) ---
        use_vllm=True,
        vllm_mode="colocate",  # 기본값이 server라 생략 가능
        vllm_gpu_memory_utilization = 0.2,
        # vllm_server_base_url="http://127.0.0.1:8000",  # ← 정확한 키

        # --- 손실/학습 관련 ---
        beta=0.0,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,  # DAPO 권고: 0.28 참고
        importance_sampling_level=args.importance_sampling_level,  # 'token'|'sequence'
        scale_rewards=(False if args.loss_type == "dr_grpo" else True),
        loss_type=args.loss_type,  # 'grpo'|'bnpo'|'dr_grpo'
        mask_truncated_completions=True,
        log_completions=True,
        wandb_log_unique_prompts =True,

        dataloader_num_workers=0,          # ★ 교착 회피 1순위
        ddp_find_unused_parameters=False,   # 미사용 파라미터 탐색 off (교착/느려짐 방지)
        ddp_backend="nccl",
    )

    print("bf16 =", training_args.bf16)
    print("eval strategy =", training_args.eval_strategy)

    print(f"train_dataset:\n{train_dataset}")
    print(train_dataset[0]['prompt'][0]['content'])
    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            check_answer,
            # penalize_english_overuse
        ],
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset if args.do_eval else None

        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        # eval_dataset = new_dataset["test"],
    )
    print("beta =", trainer.args.beta)
    
    trainer.train()

if __name__ == "__main__":
    main() 
