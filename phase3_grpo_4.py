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

    max_seq_length = 1000 # Can increase for longer reasoning traces
    lora_rank = args.lora_rank # Larger rank = smarter, but slower
    lora_alpha = args.lora_alpha # Larger rank = smarter, but slower

    model_name = args.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name = "kakaocorp/kanana-1.5-8b-instruct-2505",
        model_name = model_name, # Use Qwen3-4B-Base for 4B model
        # model_name = 'skt/A.X-4.0-Light',
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
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


    match_format = re.compile(
        rf"{args.solution_start}(.+?)$", re.DOTALL)

    # match_format = re.compile(rf".{5,}{args.solution_start}(.*?)$", re.DOTALL)


    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            match = match_format.findall(response)
            if match and ('<think>' not in match[0]) and ('</think>' not in match[0]) and ('<answer>' not in match[0]): score += 1.0
            scores.append(score)
        return scores

    def evaluate_multiple_choice(pred_answer: str, true_answer: str) -> float:
        """선다형: 보기 번호(1-5) 중 하나가 정답과 일치하면 1.0, 아니면 0.0"""
        # nums = re.findall(r"\b[1-5]\b", pred_answer)
        nums = re.findall(r"[1-5]", pred_answer)
        pred = nums[0] if nums else pred_answer.strip()
        if pred == true_answer:
            return 1.0
        return 0.0

    def evaluate_short_answer(pred_answer: str, true_answer: str) -> float:
        """단답형: 정답 후보를 '#'로 분리해서 exact match 검사"""
        for t_ans in true_answer.split("#"):
            # if pred_answer.replace(" ", "") == ans.replace(" ", ""):
            if pred_answer.strip() == t_ans:
                return 1.0
        return 0.0

    def evaluate_long_answer(pred_answer: str, true_answer: str) -> float:
        # if args.solution_start in pred_answer:
        #     return 0
        P, R, F1 = bert_score(
            [pred_answer], [true_answer],
            lang="ko",
            model_type="bert-base-multilingual-cased",
            verbose=False
        )
        return F1[0].item()

    def check_answer(prompts, completions, answer, **kwargs):
        """
        prompts:   [[{"role":"system","content":...}, {"role":"user","content":...}], ...]
        completions: [[{"content": model_output}], ...]
        answer:    [true_answer_str, ...]
        
        각 예시에 대해
        1) <answer>…</answer> 태그에서 pred_answer 추출
        2) prompts 에서 question_type: 선다형/단답형/서술형 파싱
        3) 해당 evaluate_* 호출 → 점수 리스트 반환
        """
        # 1) raw 모델 출력만 꺼내기
        responses = [c[0]["content"].strip() for c in completions]

        # 2) <answer> 태그 내부 정답만 뽑기
        preds = []
        for r in responses:
            m = match_format.search(r)
            preds.append(m.group(1).replace('</answer>','').strip() if m else "")

        scores = []
        for i, (pred, true) in enumerate(zip(preds, answer)):
            user_txt = prompts[i][-1]["content"]
            # 3) question_type 파싱
            qt_m = re.search(r"문제 유형: (선다형|단답형|서술형)", user_txt)
            qtype = qt_m.group(1).strip()

            if qtype == "선다형":
                scores.append(evaluate_multiple_choice(pred, true))
            elif qtype == "단답형":
                scores.append(evaluate_short_answer(pred, true))
            else: # "서술형"
                scores.append(evaluate_long_answer(responses[i], true))

        return scores

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
    save_name = f"grpo_v3_{model_name.split('/')[-1]}_{args.save_name}"

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
        learning_rate = 5e-6,
        weight_decay = 0.01,
        warmup_ratio = 0.03,
        beta=0,
        # lr_scheduler_type = "constant",
        optim = "adamw_8bit",
        logging_steps = 1,
        repetition_penalty = 1.1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,
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