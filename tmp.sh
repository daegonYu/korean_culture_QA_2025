#!/usr/bin/env bash


# Phase 2: Scoring
# json_path="/workspace/korean_culture_QA_2025/results/phase1_skt_A.X-4.0-Light_test_outputs.json"
# echo "Scoring: $json_path"
# answer_tag="정답:"
# python score_only_answer.py --json_path "$json_path" --answer_tag "$answer_tag"


path="/workspace/korean_culture_QA_2025/models/grpo_v3_A.X-4.0-Light_curri_선다형_단답형_v1"


# tmp 디렉토리 제외하고 그 하위 디렉토리만 순회
find "$path" -mindepth 1 -type d | while read -r dir; do
    echo "📁 디렉토리: $dir"
    # 여기에 원하는 작업 추가
done