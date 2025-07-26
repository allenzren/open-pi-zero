#!/usr/bin/env fish

##################### Paths #####################

# Set default paths
set -g DEFAULT_DATA_DIR "$PWD/data"
set -g DEFAULT_LOG_DIR "$PWD/log"

# Prompt user for input, allow override or use default
read -P "Enter the desired data directory [default: $DEFAULT_DATA_DIR], leave empty to use default: " DATA_DIR
set -g VLA_DATA_DIR (string length -- $DATA_DIR; and echo $DATA_DIR; or echo $DEFAULT_DATA_DIR)

read -P "Enter the desired logging directory [default: $DEFAULT_LOG_DIR], leave empty to use default: " LOG_DIR
set -g VLA_LOG_DIR (string length -- $LOG_DIR; and echo $LOG_DIR; or echo $DEFAULT_LOG_DIR)

# Export to current session
set -x VLA_DATA_DIR $VLA_DATA_DIR
set -x VLA_LOG_DIR $VLA_LOG_DIR

# Confirm
echo "Data directory set to: $VLA_DATA_DIR"
echo "Log directory set to: $VLA_LOG_DIR"

# Append to config.fish instead of .bashrc
echo "set -x VLA_DATA_DIR \"$VLA_DATA_DIR\"" >> ~/.config/fish/config.fish
echo "set -x VLA_LOG_DIR \"$VLA_LOG_DIR\"" >> ~/.config/fish/config.fish

echo "Environment variables VLA_DATA_DIR and VLA_LOG_DIR added to config.fish and applied to the current session."

##################### WandB #####################

read -P "Enter your WandB entity (username or team name), leave empty to skip: " ENTITY

if test -n "$ENTITY"
    set -x VLA_WANDB_ENTITY $ENTITY
    echo "WandB entity set to: $VLA_WANDB_ENTITY"
    echo "set -x VLA_WANDB_ENTITY \"$VLA_WANDB_ENTITY\"" >> ~/.config/fish/config.fish
    echo "Environment variable VLA_WANDB_ENTITY added to config.fish and applied to the current session."
else
    echo "No WandB entity provided. Please set wandb=null when running scripts to disable wandb logging and avoid error."
end

##################### HF #####################

echo "Please also set TRANSFORMERS_CACHE (Huggingface cache) and download PaliGemma weights there."
