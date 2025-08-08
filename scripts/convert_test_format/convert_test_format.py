import argparse
import json
from collections import Counter
import re

# 전체 처리
def main():
    parser = argparse.ArgumentParser(description="Score-only evaluator for test.json")

    parser.add_argument(
        "--test_file", 
        type=str, 
        help="모델이 생성한 출력 결과와 정답을 추가할 test.json 파일 경로"
    )

    parser.add_argument(
        "--reference_file", 
        type=str, 
        help="정답 후보가 포함된 참조(reference) JSON 파일의 경로"
    )

    parser.add_argument(
        "--answer_tag", 
        type=str, 
        help="정답을 추출할 때 기준이 되는 문자열 태그 (e.g., '<answer>', '정답:')"
    )

    args = parser.parse_args()

    output_file = f"{args.reference_file.replace('.json','_test_format.json')}"

    # 입력 데이터 로드 (정답 없는 쪽)
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 참조 데이터 로드 (정답 후보 포함된 쪽)
    with open(args.reference_file, "r", encoding="utf-8") as f:
        reference_all = json.load(f)

    # 모든 유형(선다형, 단답형, 서술형) 합치기
    ref_dict = {}
    for qtype in ["선다형", "단답형", "서술형"]:
        for item in reference_all.get(qtype, []):
            ref_dict[item["id"]] = item

    # 정답 추가
    for item in test_data:
        qid = item["id"]
        matched = ref_dict.get(qid)
        if matched['question_type'] == '선다형':
            item["output"] = {"answer": re.search(r'\d', matched["experiment_pred"].split(args.answer_tag)[-1].strip()).group(0)}
        else:
            item["output"] = {"answer": matched["experiment_pred"].split(args.answer_tag)[-1].replace('</answer>','').strip()}

    # 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print(f"총 {len(test_data)}개 항목 변환 완료 → {output_file}")

if __name__ == "__main__":
    main()
