import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
import re

def extract_answers(text: str) -> list:
    """
    주어진 문자열에서 <answer>...</answer> 태그 안에 있는 내용을 추출합니다.

    Args:
        text (str): 전체 텍스트 문자열

    Returns:
        List[str]: <answer> 태그 안에 있는 모든 문자열 리스트
    """
    # pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    pattern = re.compile(r"정답:(.*?)$", re.DOTALL)
    found = pattern.findall(text)
    
    if found:
        return found[0]
    return text

def extract_number_answer(pred_answer: str) -> str:
    """
    주어진 문자열에서 예측된 답변을 추출합니다.
    숫자(1-5)로 된 답변이 있는 경우 해당 숫자를 반환하고, 그렇지 않으면 전체 문자열을 반환합니다.
    
    Args:
        pred_answer (str): 예측된 답변 문자열
    
    Returns:
        str: 추출된 답변
    """
    # pred_nums = re.findall(r'\b[1-5]\b', pred_answer)
    pred_nums = re.findall(r'[1-5]', pred_answer)
    if pred_nums:
        return pred_nums[0]
    return pred_answer.strip()

# def extract_short_answer(pred_answer: str) -> str:

#     pattern = re.compile(r"'(.*?)'")
#     found = pattern.findall(pred_answer)
#     if not found:
#         return pred_answer.strip()
#     return found

def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    test_df = pd.read_json('data/test.json')
    test_df = pd.DataFrame(test_df['input'].to_list())

    # 환경 변수에 API 키 설정
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    client = OpenAI()

    # system_message = """당신은 한국의 문화에 기반하여 질문에 신뢰도 높고 정확한 답변을 생성하는 한국어 전문가 AI입니다.
    # <think></think> 태그 안에 질문에 대한 답을 하기 위해 생각을 작성하고,<answer></answer> 태그 안에 최종 답변을 작성하십시오."""
    system_message = """당신은 한국의 문화와 관련된 문제를 전문적으로 풀이해주는 문제 해설가입니다.  
사용자가 입력한 문제에 대해 정확하고 친절하게 '문제 해설'과 '정답'을 제시하세요.  
답변 형식은 반드시 다음과 같이 작성하세요:  
문제 해설: ...  
정답: ..."""

    answer_list = []
    for index, q in test_df.iterrows():
        print(f"Processing question {index + 1}/{len(test_df)}")
        
        flag = False
        if q['question_type'] == "선다형":
            format_instruction = "아래 문제를 해설해주고 정답을 알려줘."
        elif q['question_type'] == "단답형":
            # format_instruction = "5어절 이내의 **명사 또는 구**로 답하십시오. 가능한 답변이 여러 개인 경우 '#' 기호로 구분하십시오. 예: '서울#부산#대구'"
            format_instruction = "아래 문제를 해설해주고 정답을 알려줘."
        else:  # 서술형
            # format_instruction = "500자 이내로 서술하십시오."
            flag = True
            continue

        user_prompt = f"{format_instruction}\n\n문제 유형: {q['question_type']}\n문제: {q['question']}"

        n = 5
        answers = []
        # flag가 True인 경우는 서술형 문제로, 답변을 생성하지 않음
        if flag:
            answers = [""] * n
        else:
            for _ in range(n):
                response = client.responses.create(
                    model="gpt-4o",  # 또는 gpt-4o-mini 등 지원 모델
                    # tools=[{"type": "web_search"}],
                    instructions = system_message,
                    input = user_prompt
                )
                print(f"질문: {q['question']}\n답변: {response.output_text}\n")
                answers.append(response.output_text)
        # print(f"질문: {q}\n답변: {response.output_text}\n")
        answer_list.append(answers)
        print(index, end='\r')

    test_df[['answer1','answer2','answer3','answer4','answer5']] = answer_list

    test_df.to_json('data/test_with_answers_no_websearch.json', index=False,indent=4, force_ascii=False)

    for i in range(1,n+1):
        test_df[f'only_answer{i}'] = test_df[f'answer{i}'].apply(extract_answers)

    for idx,row in test_df.iterrows():
        if row['question_type'] == "선다형":
            for i in range(1,n+1):
                test_df.at[idx, f'only_answer{i}'] = extract_number_answer(row[f'only_answer{i}'])
        # elif row['question_type'] == "단답형":
        #     for i in range(1,n+1):
        #         test_df.at[idx, f'only_answer{i}'] = row[f'only_answer{i}']

    test_df.to_json('data/test_with_answers_no_websearch_.json', index=False,indent=4, force_ascii=False)

if __name__ == '__main__':
    main()
