#!/usr/bin/env bash

find results -type f -name "phase1_*_test_outputs.json" | while read -r json_path; do
  echo "Processing: $json_path"
  python scripts/score_only_answer.py --json_path "$json_path" --answer_tag ""
done