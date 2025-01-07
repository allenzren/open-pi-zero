#!/bin/bash

#SBATCH --job-name=eval-fractal
#SBATCH --output=logs/eval/%A.out
#SBATCH --error=logs/eval/%A.err
#SBATCH --time=15:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

# better to run jobs for each task
TASK_CONFIGS=(
    "google_robot_pick_horizontal_coke_can:fractal_coke"
    "google_robot_pick_vertical_coke_can:fractal_coke"
    "google_robot_pick_standing_coke_can:fractal_coke"
    "google_robot_move_near_v0:fractal_move"
    "google_robot_open_drawer:fractal_drawer"
    "google_robot_close_drawer:fractal_drawer"
    "google_robot_place_apple_in_closed_top_drawer:fractal_apple"
)
# see the config file for the number of episodes in each task

for TASK_CONFIG in "${TASK_CONFIGS[@]}" ; do

    TASK="${TASK_CONFIG%%:*}"
    CONFIG_NAME="${TASK_CONFIG##*:}"

    CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run \
        scripts/run.py \
        --config-name=$CONFIG_NAME \
        --config-path=../config/eval \
        device=cuda:0 \
        seed=42 \
        env.task=$TASK \
        horizon_steps=4 \
        act_steps=2 \
        use_bf16=False \
        use_torch_compile=True \
        name=fractal_beta \
        'checkpoint_path="...fractal_beta.pt"'
done
