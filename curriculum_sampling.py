from __init__ import *


def main():

    parser = argparse.ArgumentParser(description="Curriculum Sampling for Korean Culture QA")
    parser.add_argument("--model", required=True, type=str, help="Model name to use")
    parser.add_argument("--save_path", required=True, type=str, help="save_path")
    parser.add_argument("--system_prompt", required=True, type=str, help="system_prompt")
    parser.add_argument("--user_prompt", required=True, type=str, help="user_prompt")
    args = parser.parse_args()


    llm = LLM(
        # model='skt/A.X-4.0-Light',
        model=args.model,
        dtype="bfloat16",     # 또는 "float16", "auto"
        trust_remote_code=True,
        max_model_len=2048,   # 최대 입력 길이
        gpu_memory_utilization=0.9,  # GPU 메모리 사용률
    )

    # Sampling 파라미터 설정 (모델 로드 여부와 상관없이 준비)
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        min_p=0.1,
        top_k=-1,
        n=8
    )
    # system_prompt="""당신은 한국의 문화와 관련된 문제를 전문적으로 풀이해주는 문제 해설가입니다.  
    # 사용자가 입력한 문제에 대해 정확하고 친절하게 **문제 해설**과 **정답**을 제시하세요.  
    # 답변 형식은 반드시 다음과 같이 작성하세요:  
    # 문제 해설: ...  
    # 정답: ..."""
    # system_prompt = """당신은 한국의 문화와 관련된 문제를 전문적으로 풀이해주는 문제 해설가입니다.  
    # 사용자가 입력한 문제에 대해 정확하고 친절하게 **문제 해설**과 **정답**을 제시하세요.  
    # 답변 형식은 반드시 다음과 같이 작성하세요:  
    # 문제 해설: ...  
    # 정답: ...

    # ※ 정답 작성 형식 안내  
    # - 선다형 문제일 경우: **정답 번호**를 작성하세요.  
    # - 단답형 문제일 경우:  
    #     - 정답이 1개인 경우: 괄호, 한자 없이 **한글** 또는 **영어**를 사용하세요. (예: 사과 또는 apple)  
    #     - 정답이 여러 개인 경우: 쉼표(,)로 구분해 나열하세요. (예: 사과, 배)  
    #     - 순서 배열 문제인 경우: '-'로 구분해 정확한 순서를 유지해 나열하세요. (예: ㄱ-ㄴ-ㄷ-ㄹ)"""
    # user_prompt="아래 문제를 해설해주고 정답을 알려줘.\n반드시 문제 해설 다음에 정답을 작성해줘.\n\n카테고리: {category}\n도메인: {domain}\n키워드: {topic_keyword}\n문제 유형: {question_type}\n\n문제: {question}\n\n답변:"

    system_prompt = args.system_prompt
    user_prompt = args.user_prompt

    training_df = pd.read_csv('/workspace/korean_culture_QA_2025/data/preprocessed/grpo_train_excluded_서술형.csv')
    training_df
    results = []
    for idx, row in training_df.iterrows():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(
                # category=row['category'],
                # domain=row['domain'],
                topic_keyword=row['topic_keyword'],
                question_type=row['question_type'],
                question=row['question']
            )}
        ]

        # 생성
        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
        )
        answers = []
        for i, output in enumerate(outputs[0].outputs):
            generated_text = output.text.strip()
            answers.append(generated_text)
        print(f"Generated answer for question {row['question']}:\n{generated_text}")
        results.append([idx]+ answers)

    results_df = pd.DataFrame(results, columns=['index'] + [f'answer_{i+1}' for i in range(len(results[0])-1)])
    results_df.to_csv(args.save_path, index=False)

if __name__ == "__main__":
    main()