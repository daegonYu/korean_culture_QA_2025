
#!/bin/bash

# results 하위 모든 json 파일 중 조건에 맞는 파일만 처리
# find results -type f -name "phase1_*_test_outputs.json" | while read -r json_path; do
#   echo "Processing: $json_path"
#   python score_only_answer.py --json_path "$json_path" --answer_tag ""
# done

python score_only_answer.py --json_path "/workspace/korean_culture_QA_2025/results/phase1_grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-28_test_outputs.json" --answer_tag ""