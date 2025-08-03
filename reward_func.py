import re

match_format = re.compile(
    # rf"{args.solution_start}(.+?)$", re.DOTALL
    r"정답:(.+)"       # 태그 라인만 체크
    )
english_word_re = re.compile(r'[a-zA-Z]{2,}')   # 영어 단어 (길이 2 이상)
paren_re = re.compile(r'\([^)]*\)')    

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
    if '#' in true_answer:
        true_answer = true_answer.split('#')
        for t_ans in true_answer:
            if pred_answer.replace('*','').replace(' ','') == t_ans.replace(' ',''):
                return 1.0
    
    elif ',' in true_answer:
        true_answer = set(x.strip() for x in true_answer.split(','))
        pred_answer = set(x.replace('*','').strip() for x in pred_answer.split(','))
        if len(true_answer) == len(true_answer & pred_answer):
            return 1.0

    else:
        if pred_answer.replace('*','').replace(' ', '') == true_answer.replace(' ', ''):
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

import re


def penalize_english_overuse(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0.0
        response = completion[0]["content"]

        # 1. 괄호 안 영어 제거
        response_wo_paren = paren_re.sub('', response)

        # 2. 영어 단어 수 세기 (괄호 밖에서만)
        english_words = english_word_re.findall(response_wo_paren)

        if len(english_words) >= 3:
            score = -0.5
        scores.append(score)
    return scores
