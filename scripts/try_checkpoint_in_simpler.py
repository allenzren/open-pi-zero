import os
import random
import time

import hydra
import imageio
import numpy as np
import simpler_env
import torch
from omegaconf import OmegaConf

from src.model.vla.pizero import PiZeroInference
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time


@log_execution_time()
def load_checkpoint(model, path):
    """load to cpu first, then move to gpu"""
    data = torch.load(path, weights_only=True, map_location="cpu")
    # remove "_orig_mod." prefix if saved model was compiled
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)
    print(f"Loaded model from {path}")


def main(args):
    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # devices
    device = torch.device(f"cuda:{args.gpu_id}")

    # load default configs
    if "fractal" in args.checkpoint_path:
        cfg = OmegaConf.load(
            "config/eval/fractal_apple.yaml"
        )  # doesn't matter which task
    if "bridge" in args.checkpoint_path:
        cfg = OmegaConf.load("config/eval/bridge.yaml")

    # model
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    model = PiZeroInference(cfg, use_ddp=False)
    load_checkpoint(model, args.checkpoint_path)
    model.freeze_all_weights()
    model.to(dtype)
    model.to(device)
    if (
        args.use_torch_compile
    ):  # model being compiled in the first batch which takes some time
        model = torch.compile(
            model,
            mode="default",  # "reduce-overhead; max-autotune(-no-cudagraphs)
            # backend="inductor", # default: inductor; cudagraphs
        )
    # modes: https://pytorch.org/docs/main/generated/torch.compile.html
    # backends: https://pytorch.org/docs/stable/torch.compiler.html
    model.eval()
    print(f"Using cuda device: {device} dtype: {dtype}")
    log_allocated_gpu_memory(None, "loading model", args.gpu_id)

    # simpler env
    env = simpler_env.make(args.task)

    # env specifics
    env_adapter = hydra.utils.instantiate(cfg.env.adapter)
    env_adapter.reset()

    # run an episode
    episode_id = random.randint(0, 20)
    env_reset_options = {}
    env_reset_options["obj_init_options"] = {
        "episode_id": episode_id,  # this determines the obj inits in bridge
    }
    obs, reset_info = env.reset(options=env_reset_options)
    instruction = env.get_language_instruction()
    if args.recording:
        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false"  # avoid tokenizer forking warning about deadlock
        )
        video_writer = imageio.get_writer(f"try_{args.task}_{episode_id}.mp4")
    print(
        f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
    )
    cnt_step = 0
    inference_times = []
    while 1:
        # infer action chunk
        inputs = env_adapter.preprocess(env, obs, instruction)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(
                inputs["attention_mask"], dtype=dtype
            )
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        inputs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"].to(dtype),
            "image_text_proprio_mask": image_text_proprio_mask,
            "action_mask": action_mask,
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": inputs["proprios"].to(dtype),
        }
        inputs = {k: v.to(device) for k, v in inputs.items()}
        start_inference_time = time.time()
        with torch.inference_mode():  # speeds up
            actions = model(**inputs)
        if cnt_step > 0:
            inference_times.append(time.time() - start_inference_time)
        env_actions = env_adapter.postprocess(actions[0].float().cpu().numpy())

        # environment step
        for env_action in env_actions[: cfg.act_steps]:
            obs, reward, success, truncated, info = env.step(env_action)
            cnt_step += 1
            if truncated:
                break

        # save frame
        if args.recording:
            video_writer.append_data(env_adapter.get_video_frame(env, obs))

        # update instruction in long horizon tasks, e.g., pick apple ---> put in top drawer
        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction

        # original octo eval only done when timeout, i.e., not upon success
        if truncated:
            if args.recording:
                video_writer.close()
            break

    # summary
    print("\n\n============ Summary ============")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Action chunk steps (predicted): {cfg.horizon_steps}")
    print(f"Action chunk steps (executed): {cfg.act_steps}")
    print(f"Avg inference time (excluding first step): {np.mean(inference_times):.3f}s")
    print(
        f"Peak VRAM usage: {torch.cuda.max_memory_reserved(args.gpu_id) / 1024 ** 3:.2f} GB"
    )
    print(f"Task: {args.task}")
    print(f"Total environment steps: {cnt_step}")
    print(f"Success: {success}")
    if args.recording:
        print(f"Video saved as try_{args.task}_{episode_id}.mp4")
    print("======================================\n\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="google_robot_pick_horizontal_coke_can",
        choices=[
            "widowx_carrot_on_plate",
            "widowx_put_eggplant_in_basket",
            "widowx_spoon_on_towel",
            "widowx_stack_cube",
            "google_robot_pick_horizontal_coke_can",
            "google_robot_pick_vertical_coke_can",
            "google_robot_pick_standing_coke_can",
            "google_robot_move_near_v0",
            "google_robot_open_drawer",
            "google_robot_close_drawer",
            "google_robot_place_apple_in_closed_top_drawer",
        ],
    )
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument("--recording", action="store_true")
    args = parser.parse_args()

    # check task
    if "google_robot" in args.task:
        assert "fractal" in args.checkpoint_path
    if "widowx" in args.task:
        assert "bridge" in args.checkpoint_path

    main(args)
