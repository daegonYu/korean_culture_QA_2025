
from phase1_prompting_experiment import PromptingExperiment


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Prompting Experiment")
    parser.add_argument("--model", default="kakaocorp/kanana-1.5-8b-instruct-2505", help="Model name to use")
    parser.add_argument("--data_path", default="data", help="Path to data directory")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to test (default: all)")
    parser.add_argument("--use_train", action="store_true", help="Use train set instead of dev set")
    parser.add_argument("--use_test", action="store_true", help="Use test set instead of dev set")
    parser.add_argument("--use_lora", action="store_true", default=False, help="Use LoRA layers")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use WANDB")
    parser.add_argument("--max_lora_rank", type=int, default=64, help="Max LoRA rank (default: 64)")
    parser.add_argument("--system_prompt", type=str, default="당신은 한국 문화 전문가입니다. 질문에 답하세요.")
    parser.add_argument("--user_prompt", type=str, required=True)
    parser.add_argument("--system_prompt2", type=str, required=False, type=str, default=None, help="(Optional) Separate system prompt for descriptive (서술형) question type")
    parser.add_argument("--user_prompt2", type=str, required=False, type=str, default=None, help="(Optional) Separate user prompt for descriptive (서술형) question type")
    parser.add_argument("--answer_tag", type=str, default="정답:")

    args = parser.parse_args()
    
    print(f"🚀 Phase 1 실험 시작")
    print(f"   모델: {args.model}")
    print("="*60)
    
    if args.system_prompt2 is None:
        args.system_prompt2 = args.system_prompt

    if args.user_prompt2 is None:
        args.user_prompt2 = args.user_prompt

    experiment = PromptingExperiment(
        model_name=args.model,
        system_prompt=args.system_prompt,
        system_prompt2=args.system_prompt2,
        user_prompt=args.user_prompt,
        user_prompt2=args.user_prompt2,
        answer_tag=args.answer_tag,
        use_lora=args.use_lora,
        max_lora_rank=args.max_lora_rank if args.use_lora else None,  
        use_wandb=args.use_wandb
    )
    experiment.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    train_data, dev_data, test_data = experiment.load_data(args.data_path)

    if args.use_test:
        data_to_use = test_data
        dataset_name = "test"
        test_mode = True
    elif args.use_train:
        data_to_use = train_data
        dataset_name = "train"
        test_mode = False
    else:
        data_to_use = dev_data
        dataset_name = "dev"
        test_mode = False
    
    print(f"\n🔬 {dataset_name} set으로 실험 시작 (n={len(data_to_use)})")
    
    # 실험 실행
    results = experiment.run_experiment(data_to_use, sample_size=args.sample_size, test_mode=test_mode)

    if not test_mode:
        analysis = experiment.analyze_results(results)
        experiment.print_analysis(analysis)
        experiment.save_final_results(results, analysis)
    else:
        save_path = f"results/phase1_{'_'.join(args.model.split('/')[-2:])}_test_outputs.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 테스트셋 결과 저장 완료: {save_path}")
    
    print(f"\n✅ Phase 1 실험 완료!")
    print(f"📊 결과 파일:")
    print(f"   - phase1_detailed_results.json")
    print(f"   - phase1_analysis_summary.json")

if __name__ == "__main__":
    main() 