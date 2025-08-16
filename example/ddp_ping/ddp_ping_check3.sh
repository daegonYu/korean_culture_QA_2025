export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# SHM 작은 환경이면 일단 켭니다 (재기동 전 임시 회피)
export NCCL_SHM_DISABLE=1

export NCCL_CUMEM_ENABLE=0         # ★ 핵심: CUMEM만 끔
unset NCCL_SHM_DISABLE             # SHM 사용


torchrun --standalone --nproc_per_node=2 scripts/ddp_ping.py