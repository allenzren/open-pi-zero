# open-pi-zero

This repo implements the [pi0](https://www.physicalintelligence.company/download/pi0.pdf) model from Physical Intelligence (Pi) based on my knowledge of the paper.

The model adopts a MoE-like architecture (or the recent [MoT](https://arxiv.org/abs/2411.04996), each expert has its own set of parameters and only interacts through attention), and uses a pre-trained 3B PaliGemma VLM (2.291B to be fine-tuned) and a new set of action expert parameters (0.315B). Block-wise causal masking is used such that VLM block attends to itself, proprioception (sharing weights with action) attends to itself and VLM, and action attends to all; each block is fully bidirectional within. The model is trained with flow matching loss on the action chunk output from the action expert.

If you find a bug or think I may have misunderstood part of the architecture based on the paper, please raise an issue or email me.

## Installation
Clone this repository at your directory. If running [Simpler eval](https://github.com/simpler-env/SimplerEnv) or trying out trained checkpoints, clone my [fork](https://github.com/allenzren/SimplerEnv) (addded proprio support) to the same directory
```console
git clone https://github.com/allenzren/SimplerEnv --recurse-submodules
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run the following in the repo directory
```console
uv sync
uv pip install -e . ../SimplerEnv
uv pip install -e . ../SimplerEnv/ManiSkill2_real2sim
```
Or you may use venv or conda env instead of uv and run `pip install -e .` in all three directories.

Set environment variables `VLA_DATA_DIR` (if downloading datasets for training), `VLA_LOG_DIR`, and `VLA_WANDB_ENTITY` by running `source scripts/set_path.sh`

Download PaliGemma weights at `TRANSFORMERS_CACHE`
```console
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

### Test text generation with pre-trained weights
```console
uv run src/model/vla/pizero.py --text_only --load_pretrained_weights
```

<!-- VLA with dummy img/text, proprio, and action, output dummy flow matching action
```console
uv run src/model/vla/pizero.py
``` -->

## Try checkpoints

I have only trained with either fractal or bridge dataset so far (training with mixed OXE data soon). Links to the models:
 [Bridge-Uniform](https://huggingface.co/allenzren/open-pi-zero/blob/main/bridge_uniform.pt) | [Bridge-Gamma](https://huggingface.co/allenzren/open-pi-zero/blob/main/bridge_gamma.pt) | [Fractal-Uniform](https://huggingface.co/allenzren/open-pi-zero/blob/main/fractal_uniform.pt) | [Fractal-Gamma](https://huggingface.co/allenzren/open-pi-zero/blob/main/fractal_gamma.pt)

Uniform and Gamma stands for the schedule for sampling flow matching timesteps during training: Uniform samples uniformly between 0 and 1, and Gamma, proposed by Pi0, samples with higher density at earlier timesteps.

Run an trial in Simpler after downloading a checkpoint (see the list of tasks in the script)
```console
uv run scripts/try_checkpoint_in_simpler.py \
    --task google_robot_pick_horizontal_coke_can \
    --checkpoint_path ...fractal_gamma.pt \
    --recording
```
This roughtly takes 13.5GB of VRAM (float32).

### Training details

The models were trained with learning rate 5e-5, global batch size 1024, and roughly 22k gradient steps (not fully converged based on validation action accuracy). Input to the model includes single image (256 tokens, no history), max 20 text tokens, 1 proprio token (no history), and 4 action tokens (chunk size 4). It took roughly 2 days on one L40 node (per GPU batch size 16 and thus gradient accumulation step 8), or 12 hours with H100s. torch.compile, bfloat16, and 8-bit optimizer were used to reduce VRAM usage. Action and propriocetion data were normalized in [-1, 1].

### Inference speed

Inference involves one forward pass through PaliGemma (saving KV cache), and then 10 flow matching steps through the action expert. With RTX 4090 and float32, inference takes 230ms, which is much worse than the 73ms from the Pi0 paper (see their Table I). The main difference lies in the 10 steps through the action expert: while Pi's implementation takes only 27ms for 10 steps, mine takes 17ms per step. Flash/FlexAttention etc. might help.

### Eval results

Bridge policies run all four predicted action steps, while fractal policies run the first two steps of the predicted four.
<!-- This setup is practical given bridge data is 5Hz and fractal data is 3Hz, and the policy inference can be 4Hz. -->

Success rates in **visual matching** setting in Simpler with float32 (results in visual aggregation setting coming soon)

| Policy | Carrot on plate | Eggplant in basket | Spoon on towel | Stack cube |
|:------:|:---------------:|:------------------:|:--------------:|:----------:|
| [Bridge-Uniform](https://huggingface.co/allenzren/open-pi-zero/blob/main/bridge_uniform.pt)   | 65.3% | 86.1% | 90.3% | 18.1% |
| [Bridge-Gamma](https://huggingface.co/allenzren/open-pi-zero/blob/main/bridge_gamma.pt)    | 55.6% | 86.1% | 91.7% | 54.2% |

| Policy | Pick up Coke | Move Near | Close Drawer | Open Drawer | Open Top Drawer and Put Apple In |
|:------:|:------------:|:---------:|:------------:|:-----------:|:--------------------------------:|
| [Fractal-Uniform](https://huggingface.co/allenzren/open-pi-zero/blob/main/fractal_uniform.pt) | 91.7% | 73.8% | 79.6% | 48.1% | 64.8% |
| [Fractal-Gamma](https://huggingface.co/allenzren/open-pi-zero/blob/main/fractal_gamma.pt)    | 96.7% | 85.0% | 74.1% | 47.2% | 12.0% |

Disclaimer: please do not associate my results with possible results from Pi.

## Run training

### Download data

Download fractal data (following [OXE](https://github.com/google-deepmind/open_x_embodiment?tab=readme-ov-file))
```console
uv run gsutil -m cp -r gs://gresearch/robotics/fractal20220817_data/0.1.0   # at $VLA_DATA_DIR
```

Download bridge data from [RAIL link](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/) as suggested by OXE.

### Data pre-processing

Run (with enough RAM) [slurm/modify_rlds.sh](slurm/modify_rlds.sh), which resizes the images to 224x224 for PaliGemma. Save it at `$VLA_DATA_DIR/resize_224\`.

[Possible error](doc/error.md#5) | [Conventions on proprio / gripper actions](doc/convention.md)

### Training scripts

See examples in the [slurm](slurm/) folder. TFDS dataloading takes a growing amount of CPU RAM and roughly peaks at about 300-400GB with one dataloader in each DDP process.

[Discussion on RAM](https://github.com/openvla/openvla/issues/4) | [Possible error if running quantization](doc/error.md#9) | [My observations / lessons from training](doc/notes.md)

## Run evaluation

### Simpler

See examples in the [slurm](slurm/) folder. Currently they use the dataset statistics generated by my training; you may update `env.adapter.dataset_statistics_path` in the config to the dataset statistics json file generated in your training, located in the dataset folder.

## Things to implement / try

Tune sinusoidal embedding and RoPE parameters to better encode the relatively low number of action tokens and timestep. Multi-image (history) as input. Use EMA. Switch to GPU Simpler. Fine-tuning by adding a new expert (e.g., second camera view into pre-trained DINO/SigLIP) and gradual unmasking. Co-training with (self-)supervision on modalities other than action.

## Acknowledgement

PaliGemma setup is largely adopted from [Open-source PaliGemma](https://github.com/hkproj/pytorch-paligemma/tree/main). Dataset loading is adopted from [Octo](https://octo-models.github.io/) and [dlimp](https://github.com/kvablack/dlimp). Dataset pre-processing is adopted from [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod/tree/main). Other references: [Pi0](https://www.physicalintelligence.company/download/pi0.pdf), [OpenVLA](https://github.com/openvla/openvla), [Flow matching](https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb), [Implementation by lucidrains](https://github.com/lucidrains/pi-zero-pytorch), [SimplerEnv](https://github.com/simpler-env/SimplerEnv).

Special thanks to [Asher Hancock](https://aasherh.github.io/) for the discussion on block-wise causal masking.
