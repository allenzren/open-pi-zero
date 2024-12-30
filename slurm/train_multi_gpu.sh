#!/bin/bash

#SBATCH --job-name=pg-vla
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --time=47:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G

# GPU Check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)

# Run script with selected configuration using torchrun
HYDRA_FULL_ERROR=1 uv run torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPU \
  --rdzv_id=100 \
  --rdzv_backend=c10d \
  --max-restarts=1 \
  --standalone \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  scripts/run.py \
  --config-name=bridge \
  action_lr=0.00005 \
  vlm_lr=0.00005 \
  flow_schedule=gamma