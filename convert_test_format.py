import json
from collections import Counter
import re

# 경로 설정
input_file = "/workspace/korean_culture_QA_2025/data/test.json"
# answer 포함된 예측 결과
reference_file = "/workspace/korean_culture_QA_2025/results/phase1_grpo_v4_A.X-4.0-Light_curri_선다형_단답형_v2_prompt2_checkpoint-104_test_outputs.json"
answer_tag = '정답:'
# answer_tag = '<answer>'
output_file = f"{reference_file.replace('.json','_test_format.json')}"


# 전체 처리
def main():
    # 입력 데이터 로드 (정답 없는 쪽)
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # 참조 데이터 로드 (정답 후보 포함된 쪽)
    with open(reference_file, "r", encoding="utf-8") as f:
        reference_all = json.load(f)

    # 모든 유형(선다형, 단답형, 서술형) 합치기
    ref_dict = {}
    for qtype in ["선다형", "단답형", "서술형"]:
        for item in reference_all.get(qtype, []):
            ref_dict[item["id"]] = item

    # 정답 추가
    for item in input_data:
        qid = item["id"]
        matched = ref_dict.get(qid)
        if matched['question_type'] == '선다형':
            item["output"] = {"answer": re.search(r'\d', matched["experiment_pred"].split(answer_tag)[-1].strip()).group(0)}
        else:
            item["output"] = {"answer": matched["experiment_pred"].split(answer_tag)[-1].replace('</answer>','').strip()}

    # 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(input_data, f, indent=4, ensure_ascii=False)

    print(f"총 {len(input_data)}개 항목 변환 완료 → {output_file}")

if __name__ == "__main__":
    main()
