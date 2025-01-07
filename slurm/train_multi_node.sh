#!/bin/bash

#SBATCH --job-name=pg-vla
#SBATCH --output=logs/%A-%N.out
#SBATCH --error=logs/%A-%N.err
#SBATCH --time=47:59:59
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G  # per node

export WANDB__SERVICE_WAIT=300

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)
echo Node IP: $head_node_ip
echo Master Port: $MASTER_PORT

# export LOGLEVEL=INFO
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Run script with selected configuration using torchrun
# TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=INFO TORCH_SHOW_CPP_STACKTRACES=1 NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=ens27f0 srun uv run torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=8 \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --max-restarts=3 \
  --rdzv_endpoint $head_node_ip:$MASTER_PORT \
  scripts/run.py \
  --config-name=bridge \
  n_nodes=$SLURM_JOB_NUM_NODES \
  action_lr=0.00005 \
  vlm_lr=0.00005 \
  flow_sampling=beta \
  use_torch_compile=True \
  use_bf16=True
