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
from peft import PeftConfig
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import wandb


warnings.filterwarnings('ignore')
load_dotenv()


class PromptingExperiment:
    def __init__(self, model_name="beomi/Kanana-8B", load_model=True, use_lora=False, use_wandb=False, \
                system_prompt="", system_prompt2="", user_prompt="", user_prompt2="", answer_tag="정답:", max_lora_rank=64):
        """
        Phase 1: 프롬프팅 실험
        5가지 다른 프롬프트로 Kanana 8B 성능 테스트
        """

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.system_prompt2 = system_prompt2        # 서술형 유형에 대한 시스템 프롬프트 (서술형의 경우 따로 프롬프팅)
        self.user_prompt = user_prompt
        self.user_prompt2 = user_prompt2            # 서술형 유형에 대한 유저 프롬프트
        self.answer_tag = answer_tag
        self.use_lora = use_lora
        self.use_wandb = use_wandb
        self.max_lora_rank = max_lora_rank
        # wandb 초기화
        if self.use_wandb:
            load_dotenv()
            wandb_api_key = os.getenv("WANDB_API_KEY")

            wandb.login(key=wandb_api_key)

            wandb.init(
                project="moducorpus_korea_culture_answer_log",
                name=f"{model_name.replace('/', '_')}",
                config={
                    "model_name": model_name,
                    "load_model": load_model,
                    "use_lora": self.use_lora,
                    "system_prompt": system_prompt,
                    "system_prompt_서술형": system_prompt2,
                    "user_prompt_template": user_prompt,
                    "user_prompt_template_서술형": user_prompt2,
                    "answer_tag": answer_tag
                }
            )
            self.wb_table = wandb.Table(columns=["prompt_system", "prompt_user", "answer"])

        self.llm = None
        self.lora_req = None
        max_model_len=3096
        if load_model and not self.use_lora:
            print(f"Loading {self.model_name}...")
            self.llm = LLM(
                model=self.model_name,
                dtype="bfloat16",     # 또는 "float16", "auto"
                trust_remote_code=True,
                max_model_len=max_model_len,   # 최대 입력 길이
                gpu_memory_utilization=0.9,  # GPU 메모리 사용률
            )
        elif load_model and self.use_lora:
            config = PeftConfig.from_pretrained(self.model_name)
            base_model_name = config.base_model_name_or_path
            print(f"Loading {self.model_name} with LoRA Layers...")
            self.llm = LLM(
                model=base_model_name,
                dtype="bfloat16",     # 또는 "float16", "auto"
                trust_remote_code=True,
                max_model_len=max_model_len,   # 최대 입력 길이
                gpu_memory_utilization=0.9,  # GPU 메모리 사용률
                enable_lora=self.use_lora,
                max_lora_rank=self.max_lora_rank
            )
            self.lora_req = LoRARequest("adapter", 1, self.model_name)

        # Sampling 파라미터 설정 (모델 로드 여부와 상관없이 준비)
        self.sampling_params = SamplingParams(
            max_tokens=2000,
            temperature=0.6,
            top_p=0.95,
            # best_of=8,
            # n=1
        )
                
    def load_data(self, data_path="data"):
        """데이터 로드"""
        data_dir = Path(data_path)
        
        with open(data_dir / "train.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(data_dir / "dev.json", "r", encoding="utf-8") as f:
            dev_data = json.load(f)
        with open(data_dir / "preprocessed/test.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        print(f"Loaded {len(train_data)} train, {len(dev_data)} dev" + (f", {len(test_data)} test" if test_data else ""))
        return train_data, dev_data, test_data
    
    def create_prompts(self, sample):
        """5가지 프롬프트 생성"""
        question = sample['input']['question']
        question_type = sample['input']['question_type']
        category = sample['input']['category']
        domain = sample['input']['domain']
        topic_keyword = sample['input']['topic_keyword']
        
        prompts = {}

        if question_type != '서술형':
            system_prompt = self.system_prompt
            user_prompt = self.user_prompt
            user_prompt = user_prompt.format(
                category=category,
                domain=domain,
                topic_keyword=topic_keyword,
                question_type=question_type,
                question=question
            )
        else:
            system_prompt = self.system_prompt2
            user_prompt = self.user_prompt2
            user_prompt = user_prompt.format(
                category=category,
                domain=domain,
                topic_keyword=topic_keyword,
                question_type=question_type,
                question=question
            )
        print(f"system_prompt:{system_prompt}")    
        print(f"user_prompt:{user_prompt}")

        prompts['experiment'] = {"system_prompt":system_prompt, "user_prompt": user_prompt}
        
        return prompts
    
    def generate_answer(self, prompt):
        """모델로 답변 생성"""
        try:
            messages = [
                {"role": "system", "content": prompt['system_prompt']},
                {"role": "user", "content": prompt['user_prompt']}
            ]

            # 생성
            if self.use_lora:
                outputs = self.llm.chat(
                    messages=messages,
                    sampling_params=self.sampling_params,
                    lora_request=self.lora_req
                )
            else:
                outputs = self.llm.chat(
                    messages=messages,
                    sampling_params=self.sampling_params,
                )

            # 디코딩 (입력 길이 이후만 추출)
            generated_text = outputs[0].outputs[0].text
            
            # 답변 정리
            answer = generated_text.strip()

            # wandb 로그 추가
            if self.use_wandb:
                self.wb_table.add_data(
                    prompt['system_prompt'],
                    prompt['user_prompt'],
                    answer
                )

            print(answer)
            return answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def evaluate_multiple_choice(self, pred_answer, true_answer):
        """선다형 평가"""
        # 숫자 추출
        # pred_nums = re.findall(r'\b[1-5]\b', pred_answer)
        pred_nums = re.findall(r'[1-5]', pred_answer)
        if pred_nums:
            pred = pred_nums[0]
        else:
            pred = pred_answer.strip()
        
        return 1 if pred == true_answer else 0
    
    def evaluate_short_answer(self, pred_answer, true_answer):
        """
        Calculate Exact Match score where true_data may contain multiple acceptable answers separated by #
        """
        correct = 0
        true_answer_list = true_answer.split('#')

        # if any(pred_answer.replace(' ','') == ans.replace(' ','') for ans in true_answer_list):
        if any(pred_answer == true_answer for true_answer in true_answer_list):
            correct = 1
                
        return correct
    
    def evaluate_long_answer(self, pred_answer, true_answer):
        """서술형 평가 (ROUGE-1, ROUGE-2, ROUGE-L via rouge_metric)"""
        from rouge_metric import Rouge

        rouge_evaluator = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1000,
            length_limit_type="words",
            use_tokenizer=True,
            apply_avg=True,
            apply_best=False,
            alpha=0.5,        # F1 score
            weight_factor=1.0,
        )

        # 단일 문장 pair도 리스트로 감싸서 전달
        scores = rouge_evaluator.get_scores(
            [pred_answer],
            [true_answer]
        )

        return {
            'rouge1':   scores['rouge-1']['f'],
            'rouge2':   scores['rouge-2']['f'],
            'rougeL':   scores['rouge-l']['f']
        }

    def calc_BLEU(self, pred_answer: str, true_answer: str, apply_avg=True, apply_best=False, use_mecab=True):
        from nltk.translate.bleu_score import sentence_bleu
        from konlpy.tag import Mecab

        tokenizer = Mecab()
        stacked_bleu = []

        # 1) 단일 문자열을 참조 리스트로 감싸기
        refs = [true_answer]  # 리스트 of reference strings
        cand = pred_answer    # 하나의 candidate string

        # 2) 토크나이징
        if use_mecab:
            cand_tokens = tokenizer.morphs(cand)
        else:
            cand_tokens = cand.split()

        # 3) 각 ref 텍스트별로 BLEU 계산
        best_bleu = 0
        sum_bleu  = 0
        for ref_text in refs:
            if use_mecab:
                ref_tokens = tokenizer.morphs(ref_text)
            else:
                ref_tokens = ref_text.split()

            score = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0))
            sum_bleu  += score
            best_bleu = max(best_bleu, score)

        # 4) 평균
        avg_bleu = sum_bleu / len(refs)  # 이제 len(refs) == 1

        # 5) 결과 수집
        if apply_best:
            stacked_bleu.append(best_bleu)
        if apply_avg:
            stacked_bleu.append(avg_bleu)

        return sum(stacked_bleu) / len(stacked_bleu) if stacked_bleu else 0.0


    def run_experiment(self, data, sample_size=None, save_results=True, test_mode=False):
        """실험 실행"""
        if sample_size:
            data = data[:sample_size]
            print(f"Using {sample_size} samples for testing")
        
        results = {
            '선다형': [],
            '단답형': [],
            '서술형': []
        }
        
        for i, sample in enumerate(tqdm(data, desc="Running experiments")):
            question_type = sample['input']['question_type']
            true_answer = sample['output']['answer'] if 'output' in sample else None
            
            # 5가지 프롬프트 생성
            prompts = self.create_prompts(sample)
            
            sample_results = {
                'id': sample['id'],
                'question_type': question_type,
                'true_answer': true_answer,
                'question': sample['input']['question']
            }
            
            # 각 프롬프트로 답변 생성 및 평가
            for prompt_name, prompt in prompts.items():
                pred_answer = self.generate_answer(prompt=prompt)
                sample_results[f'{prompt_name}_pred'] = pred_answer
                if test_mode:
                    continue  # 평가 스킵

                original_answer = pred_answer
                if self.answer_tag != '' and self.answer_tag in pred_answer:
                    pred_answer = pred_answer.split(self.answer_tag)[-1].strip()
                pred_answer = pred_answer.replace('*', '').replace('</answer>','')
                
                # 질문 유형별 평가
                if question_type == "선다형":
                    score = self.evaluate_multiple_choice(pred_answer, true_answer)
                    sample_results[f'{prompt_name}_score'] = score
                    
                elif question_type == "단답형":
                    exact = self.evaluate_short_answer(pred_answer, true_answer)
                    sample_results[f'{prompt_name}_exact'] = exact
                    
                else:  # 서술형
                    # 1) Rouge
                    rouge_scores = self.evaluate_long_answer(original_answer, true_answer)
                    sample_results[f'{prompt_name}_rouge1'] = rouge_scores['rouge1']
                    sample_results[f'{prompt_name}_rouge2'] = rouge_scores['rouge2']
                    sample_results[f'{prompt_name}_rougeL'] = rouge_scores['rougeL']

                    # 2) BLEU
                    bleu_score = self.calc_BLEU(original_answer, true_answer)
                    sample_results[f'{prompt_name}_bleu'] = bleu_score
            
            results[question_type].append(sample_results)
            
            # 중간 저장 (10개마다)
            # if ((i + 1) % 100 == 0) or (i + 1 == len(data)) and save_results:
            #     self.save_intermediate_results(results, i + 1)
        
        wandb.log({"prompts_and_answers": self.wb_table})
        return results
    
    def save_intermediate_results(self, results, current_idx):
        """중간 결과 저장"""
        save_path = f"results/phase1_{self.model_name.split('/')[-1]}_intermediate_results_{current_idx}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def only_scoring(self, data: dict) -> dict:
        """
        data: {
        "선다형": [ { "id":..., "question_type":"선다형", "true_answer":..., "question":..., 
                    "rich_pred":..., "format_aware_pred":..., ... }, … ],
        "단답형": [ … ],
        "서술형": [ … ]
        }
        """
        results = {
            '선다형': [],
            '단답형': [],
            '서술형': []
        }

        for question_type, samples in data.items():
            for sample in samples:
                rec = sample.copy()  # 원본 레코드 복사

                # `_pred` 로 끝나는 필드만 찾아서 평가
                for key, pred_answer in sample.items():
                    if not key.endswith('_pred'):
                        continue

                    name = key[:-5]  # 'rich_pred' -> 'rich'

                    original_answer = pred_answer
                    if '<answer>' in pred_answer:
                        answer_tag = '<answer>'
                    else:
                        answer_tag = '정답:'
                    pred_answer = pred_answer.split(answer_tag)[-1].strip()
                    pred_answer = pred_answer.replace('*', '').replace('</answer>','')

                    if rec['true_answer'] is None:
                        print('# 불량')
                        print(rec)

                    # 질문 유형별 평가
                    if question_type == "선다형":
                        rec[f'{name}_score'] = self.evaluate_multiple_choice(pred_answer, rec['true_answer'])
                        
                    elif question_type == "단답형":
                        rec[f'{name}_exact'] = self.evaluate_short_answer(pred_answer, rec['true_answer'])
                        
                    else:  # 서술형
                        # 1) Rouge
                        rouge = self.evaluate_long_answer(original_answer, rec['true_answer'])
                        rec[f'{name}_rouge1'] = rouge['rouge1']
                        rec[f'{name}_rouge2'] = rouge['rouge2']
                        rec[f'{name}_rougeL'] = rouge['rougeL']

                        # 2) BLEU
                        rec[f'{name}_bleu']   = self.calc_BLEU(original_answer, rec['true_answer'])


                results[question_type].append(rec)

        return results


    def analyze_results(self, results):
        """결과 분석 및 요약"""
        analysis = {}
        
        for question_type in ['선다형', '단답형', '서술형']:
            if not results[question_type]:
                continue
                
            type_data = results[question_type]
            analysis[question_type] = {}
            
            for prompt_name in ['_'.join(x.split('_')[:-1]) for x in type_data[0].keys() if x.endswith('_pred')]:
                if question_type == "선다형":
                    print(type_data[0].keys())
                    scores = [item[f'{prompt_name}_score'] for item in type_data]
                    analysis[question_type][prompt_name] = {
                        'accuracy': np.mean(scores),
                        'count': len(scores)
                    }
                    
                elif question_type == "단답형":
                    exact_scores = [item[f'{prompt_name}_exact'] for item in type_data]
                    analysis[question_type][prompt_name] = {
                        'exact_match': np.mean(exact_scores),
                        'count': len(exact_scores)
                    }
                    
                else:  # 서술형
                    rouge1_scores = [item[f'{prompt_name}_rouge1'] for item in type_data]
                    rouge2_scores = [item[f'{prompt_name}_rouge2'] for item in type_data]
                    rougeL_scores = [item[f'{prompt_name}_rougeL'] for item in type_data]
                    bleu_scores     = [item[f'{prompt_name}_bleu']      for item in type_data]
                    analysis[question_type][prompt_name] = {
                        'rouge1': np.mean(rouge1_scores),
                        'rouge2': np.mean(rouge2_scores),
                        'rougeL': np.mean(rougeL_scores),
                        'bleu': np.mean(bleu_scores),
                        'count': len(rouge1_scores)
                    }
        
        return analysis
    
    def print_analysis(self, analysis, save_path="analysis_result.md"):
        """분석 결과 출력 및 .md 파일 저장"""
        output_lines = []

        def write(line=""):
            print(line)
            output_lines.append(line)

        write("\n" + "="*80)
        write("PHASE 1: PROMPTING EXPERIMENT RESULTS")
        write("="*80)
        
        for question_type, type_results in analysis.items():
            write(f"\n📊 {question_type} 결과:")
            write("-" * 50)
            
            if question_type == "선다형":
                for prompt_name, metrics in type_results.items():
                    write(f"{prompt_name:15}: Accuracy = {metrics['accuracy']:.3f} (n={metrics['count']})")
                    
            elif question_type == "단답형":
                for prompt_name, metrics in type_results.items():
                    write(f"{prompt_name:15}: Exact = {metrics['exact_match']:.3f} (n={metrics['count']})")
                    
            else:  # 서술형
                for prompt_name, metrics in type_results.items():
                    write(
                        f"{prompt_name:15}: "
                        f"ROUGE-1 = {metrics['rouge1']:.3f}, "
                        f"ROUGE-2 = {metrics['rouge2']:.3f}, "
                        f"ROUGE-L = {metrics['rougeL']:.3f}, "
                        f"BLEU = {metrics['bleu']:.3f} "
                        f"(n={metrics['count']})"
                    )

        write(f"\n🏆 종합 순위:")
        write("-" * 30)

        prompt_scores = {}
        for prompt_name in analysis['선다형'].keys():
            total_score = 0
            total_count = 0
            
            for question_type, type_results in analysis.items():
                if prompt_name in type_results:
                    if question_type == "선다형":
                        score = type_results[prompt_name]['accuracy']
                    elif question_type == "단답형":
                        score = type_results[prompt_name]['exact_match']
                    else:  # 서술형
                        score = type_results[prompt_name]['rouge1']
                    
                    count = type_results[prompt_name]['count']
                    total_score += score * count
                    total_count += count
            
            if total_count > 0:
                prompt_scores[prompt_name] = total_score / total_count

        # 순위 정렬 및 출력
        ranked_prompts = sorted(prompt_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (prompt_name, score) in enumerate(ranked_prompts, 1):
            write(f"{i}. {prompt_name:15}: {score:.3f}")
        
        # .md 파일 저장
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))


    def save_final_results(self, results, analysis):
        """최종 결과 저장"""
        # 상세 결과
        with open(f'results/phase1_{"_".join(self.model_name.split("/")[-2:])}_detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 분석 요약
        with open(f'results/phase1_{"_".join(self.model_name.split("/")[-2:])}_analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 결과 저장 완료:")
        print(f"   - phase1_{'_'.join(self.model_name.split('/')[-2:])}_detailed_results.json")
        print(f"   - phase1_{'_'.join(self.model_name.split('/')[-2:])}_analysis_summary.json")

def main():
    """메인 실행 함수"""
    print("🚀 Phase 1")
    print("="*60)
    
    # 실험 초기화
    experiment = PromptingExperiment()
    
    # 데이터 로드
    train_data, dev_data, test_data = experiment.load_data()
    
    # Dev set으로 실험 (시간 절약을 위해)
    print(f"\n🔬 Dev set으로 실험 시작 (n={len(dev_data)})")
    
    # 실험 실행
    results = experiment.run_experiment(dev_data, sample_size=None)
    
    # 결과 분석
    analysis = experiment.analyze_results(results)
    
    # 결과 출력
    experiment.print_analysis(analysis)
    
    # 결과 저장
    experiment.save_final_results(results, analysis)
    
    print(f"\n✅ Phase 1 실험 완료!")

if __name__ == "__main__":
    main() 