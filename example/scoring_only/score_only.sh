#!/usr/bin/env bash

json_path="results/phase1_grpo_v2_A.X-4.0-Light_선다형_단답형_checkpoint-56_test_outputs.json"
echo "Processing: $json_path"
python scoring_only/score_only_answer.py --json_path "$json_path" --answer_tag ""