"""
To run this: 

open-pi-zero$ python scripts/open_pi0_server.py  --checkpoint_path bridge_beta_step19296_2024-12-26_22-30_42.pt    --use_bf16     --use_torch_compile
"""

import argparse
import random
import time

import torch
import numpy as np
from omegaconf import OmegaConf
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import uvicorn
import os
from PIL import Image


# ruff: noqa: E402
import json
import json_numpy

json_numpy.patch()

# If your code references additional modules (like simpler_env, custom adapters, etc.), import them:
# from src.model.vla.pizero import PiZeroInference
# from src.utils.monitor import log_execution_time, log_allocated_gpu_memory
# import simpler_env
# import hydra

SAMPLE_PAYLOAD = {
    "instruction": "Pick up the apple",
    "image": np.zeros((256, 256, 3), dtype=np.uint8),
    "proprio": np.ones(7),
    "unnorm_key": "bridge_orig",
}


###############################################################################
# Helper Function
###############################################################################
def load_checkpoint(model, path):
    """Load checkpoint to CPU first, then move to GPU. 
       Adjusts compiled model keys (removing '_orig_mod.').
    """
    data = torch.load(path, map_location="cpu")
    # If your checkpoint includes "weights_only=True", adjust below:
    if "model" in data:  # typical format
        state_dict = data["model"]
    else:
        state_dict = data
    # remove "_orig_mod." prefix if the saved model was compiled
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f"[PiZeroServer] Loaded model from {path}")


###############################################################################
# PiZeroServer Class
###############################################################################
class PiZeroServer:
    def __init__(
        self,
        checkpoint_path: str,
        gpu_id: int = 0,
        use_bf16: bool = False,
        use_torch_compile: bool = False,
    ):
        """Initialize the PiZero server (model loading, device setup, etc.)."""
        # seeding
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.checkpoint_path = checkpoint_path

        # --------------------------------------------------------------------
        # (1) Load a default config (fractal vs. bridge) depending on checkpoint name
        # --------------------------------------------------------------------
        if "fractal" in checkpoint_path:
            cfg_path = "config/eval/fractal_apple.yaml"
        elif "bridge" in checkpoint_path:
            cfg_path = "config/eval/bridge.yaml"
        else:
            raise ValueError(
                f"Checkpoint path '{checkpoint_path}' must contain 'fractal' or 'bridge' in its name "
                "to determine the default config file. Please adjust logic as needed."
            )

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Could not find config file {cfg_path}. Update path or logic to match your local usage."
            )
        cfg = OmegaConf.load(cfg_path)

        # If your checkpoint name also determines the flow sampling schedule:
        if "uniform" in checkpoint_path:
            cfg.flow_sampling = "uniform"
        if "beta" in checkpoint_path:
            cfg.flow_sampling = "beta"

        self.cfg = cfg

        # --------------------------------------------------------------------
        # (2) Build PiZeroInference model and load checkpoint
        # --------------------------------------------------------------------
        # from src.model.vla.pizero import PiZeroInference
        from src.model.vla.pizero import PiZeroInference  # keep it local if needed

        self.model = PiZeroInference(cfg, use_ddp=False)
        load_checkpoint(self.model, checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        if use_torch_compile:
            self.model = torch.compile(self.model, mode="default")
            print("[PiZeroServer] Using torch.compile() on PiZero model.")

        self.model.eval()
        print(f"[PiZeroServer] Ready on device={self.device} with dtype={self.dtype}")

        # --------------------------------------------------------------------
        # (3) [Optional] Build environment adapter if you want to unify logic
        # --------------------------------------------------------------------
        # Typically done with Hydra's `env_adapter = hydra.utils.instantiate(cfg.env.adapter)`
        # But if you prefer a simpler approach, skip or implement your custom preprocessing here.
        self.env_adapter = None
        import hydra
        env_adapter = hydra.utils.instantiate(cfg.env.adapter)
        self.env_adapter = env_adapter
        print("[PiZeroServer] Successfully instantiated environment adapter.")

    def predict_action_chunk(self, obs_dict: dict, instruction: str):
        """
        Takes an observation dict + instruction, runs PiZero inference,
        and returns the predicted action chunk as a numpy array.
        
        Example `obs_dict` might contain:
            {
              "image": (H x W x 3) image 
              "proprio": [joint_positions, gripper, etc.],
              ...
            }
        
        Adjust to your actual environment / input format.
        """
        # -- 1) Preprocess into model's input format
        # If you have an adapter:
        if self.env_adapter is not None:
            # This typically expects something like `obs`, `instruction`, and returns tokenized input
            # Because we do not have an actual environment object here, you might override the adapter's usage.
            # Example:
            #    inputs = self.env_adapter.preprocess(None, obs_dict, instruction)
            # But you'll need to confirm that your adapter doesn't require the real gym env instance.
            # try:
            # check if obs_dict contains "image" and "proprio" keys
            assert "image" in obs_dict and "proprio" in obs_dict
            inputs = self.env_adapter.preprocess(None, obs_dict, instruction)
            # except:
            #     raise ValueError(
            #         "env_adapter.preprocess() failed; adapt your code so that it can handle raw data."
            #     )
        else:
            raise NotImplementedError(
                "No env_adapter found. Implement custom data preprocessing here."
            )

        # Build causal masks
        with torch.no_grad():
            (
                causal_mask,
                vlm_position_ids,
                proprio_position_ids,
                action_position_ids,
            ) = self.model.build_causal_mask_and_position_ids(
                inputs["attention_mask"], dtype=self.dtype
            )
            image_text_proprio_mask, action_mask = self.model.split_full_mask_into_submasks(causal_mask)

            # -- 2) Put everything on device
            model_inputs = {
                "input_ids": inputs["input_ids"].to(self.device),
                "pixel_values": inputs["pixel_values"].to(self.device, dtype=self.dtype),
                "image_text_proprio_mask": image_text_proprio_mask.to(self.device),
                "action_mask": action_mask.to(self.device),
                "vlm_position_ids": vlm_position_ids.to(self.device),
                "proprio_position_ids": proprio_position_ids.to(self.device),
                "action_position_ids": action_position_ids.to(self.device),
                "proprios": inputs["proprios"].to(self.device, dtype=self.dtype),
            }

            # -- 3) Forward pass
            actions = self.model(**model_inputs)

            # actions is typically shape (batch=1, horizon_steps, action_dim)
            # We might return the first chunk of predicted actions.
            actions = self.env_adapter.postprocess(actions[0].float().cpu().numpy())
            return actions


###############################################################################
# FastAPI Setup
###############################################################################
app = FastAPI()
server: PiZeroServer = None


@app.on_event("startup")
def load_model_on_startup():
    """Loads the PiZeroServer once when the server starts (rather than every request)."""
    global server
    # You can parse environment variables or keep them fixed:
    checkpoint = os.environ.get("OPEN_PIZERO_CKPT", "path/to/pizero_checkpoint.pt")
    gpu_id = int(os.environ.get("OPEN_PIZERO_GPU_ID", 0))
    use_bf16 = bool(int(os.environ.get("OPEN_PIZERO_USE_BF16", 0)))
    use_torch_compile = bool(int(os.environ.get("OPEN_PIZERO_TORCH_COMPILE", 0)))

    print("[Server Startup] Loading PiZeroServer with config:")
    print(f"  checkpoint_path = {checkpoint}")
    print(f"  gpu_id          = {gpu_id}")
    print(f"  use_bf16        = {use_bf16}")
    print(f"  use_torch_compile = {use_torch_compile}")
    server = PiZeroServer(
        checkpoint_path=checkpoint,
        gpu_id=gpu_id,
        use_bf16=use_bf16,
        use_torch_compile=use_torch_compile,
    )
    
    print("Passing sample to policy for JIT compilation")
    action = server.predict_action_chunk(SAMPLE_PAYLOAD, SAMPLE_PAYLOAD["instruction"])
    print(f"action shape: {action.shape}")


@app.post("/act")
def act(
    payload: dict = Body(
        ...,
        example={
            "image": [...],  # image data
            "proprio": [...],  # proprioception data
            "instruction": "Pick up the apple",
            "unnorm_key": "bridge_orig",  # optional key for de-normalization
        }
    )
):
    """Predict an action chunk from a single observation + instruction."""
    global server
    
    if double_encode := "encoded" in payload:
        # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
        assert len(payload.keys()) == 1, "Only uses encoded payload!"
        payload = json.loads(payload["encoded"])
    
    print("[Server] Received POST request at /act")
    if server is None:
        return {"error": "Server not ready; model failed to load."}

    instruction = payload["instruction"]
    image_data = payload["image"]
    # Ensure we have a proper uint8 array
    # array = np.array(image_data).astype(np.uint8)
    payload["image"] = np.array(image_data)
    payload["proprio"] = np.array(payload["proprio"])

    # Predict an action chunk
    predicted_actions = server.predict_action_chunk(payload, instruction)
    if double_encode:
        return JSONResponse(json_numpy.dumps(predicted_actions))
    else:
        return JSONResponse(predicted_actions)


###############################################################################
# Main Entry Point
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to PiZero checkpoint .pt")
    parser.add_argument("--gpu_id", type=int, default=0, help="Which GPU ID to use")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 instead of float32")
    parser.add_argument("--use_torch_compile", action="store_true", help="Use torch.compile() on model forward pass")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    # Environment variables read by `load_model_on_startup()`
    os.environ["OPEN_PIZERO_CKPT"] = args.checkpoint_path
    os.environ["OPEN_PIZERO_GPU_ID"] = str(args.gpu_id)
    os.environ["OPEN_PIZERO_USE_BF16"] = "1" if args.use_bf16 else "0"
    os.environ["OPEN_PIZERO_TORCH_COMPILE"] = "1" if args.use_torch_compile else "0"

    # Launch uvicorn server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
