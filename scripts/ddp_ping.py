# ddp_ping.py
import os, torch, torch.distributed as dist, datetime, sys
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)  # ★ init PG 전에 고정
print(f"[rank {os.environ['RANK']}] set_device -> {torch.cuda.current_device()}", flush=True)

dist.init_process_group("nccl", init_method="env://",
                        timeout=datetime.timedelta(seconds=180))
print(f"[rank {os.environ['RANK']}] PG init OK", flush=True)

x = torch.tensor([1.0], device="cuda")
dist.all_reduce(x)
print(f"[rank {os.environ['RANK']}] all_reduce OK x={x.item()}", flush=True)

dist.destroy_process_group()
print(f"[rank {os.environ['RANK']}] done", flush=True)
