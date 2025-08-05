# 모두 말뭉치 제출 파일 형식 일치 도구

이 도구는 생성된 모델의 출력 결과 파일(test.json)에 정답을 추가하여 모두 말뭉치 제출 파일 형식으로 변환하는 용도로 사용됩니다.

## 사용법

```bash
python convert_test_format/convert_test_format.py --test_file ./data/test.json --reference_file path/to/reference.json --answer_tag "<answer>"
```

### Arguments 설명

- `--test_file`: 모델이 생성한 출력 결과와 정답을 추가할 `test.json` 파일 경로
- `--reference_file`: 정답 후보가 포함된 참조(reference) JSON 파일 경로
- `--answer_tag`: 정답을 추출할 때 기준이 되는 문자열 태그 (예: `<answer>`, `정답:`)

### 예시

```bash
python convert_test_format/convert_test_format.py --test_file ./data/test.json --reference_file ./data/reference.json --answer_tag "정답:"
```

### 출력 결과

실행 결과로 `{reference_file}_test_format.json` 형태의 파일이 생성됩니다.

```
총 {n}개 항목 변환 완료 → {reference_file}_test_format.json
```

생성된 파일을 제출 시 사용하면 됩니다.
