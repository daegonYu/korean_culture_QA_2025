trl vllm-serve --model trillionlabs/Tri-7B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 1024 --dtype bfloat16 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.5 \
  --trust-remote-code