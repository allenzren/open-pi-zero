import json
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoTokenizer

try:
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
except ImportError:
    # NOTE(YL): make simpler_env optional so we can support non-simplerenv tasks
    print("Failed to simpler_env, we will ignore the error for now")
    pass

from src.agent.env_adapter.base import BaseEnvAdapter
from src.model.vla.processing import VLAProcessor
from src.utils.geometry import euler2axangle, mat2euler, quat2mat


class SimplerAdapter(BaseEnvAdapter):
    def __init__(
        self,
        dataset_statistics_path: str,
        pretrained_model_path: str,
        tokenizer_padding: str,
        num_image_tokens: int,
        image_size: Tuple[int, int],
        max_seq_len: int,
        action_normalization_type: str = "bound",
        proprio_normalization_type: str = "bound",
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.action_normalization_type = action_normalization_type
        self.proprio_normalization_type = proprio_normalization_type
        assert action_normalization_type in ["bound", "gaussian"]
        assert proprio_normalization_type in ["bound", "gaussian"]

        # for normalization
        with tf.io.gfile.GFile(dataset_statistics_path, "r") as f:
            self.dataset_statistics = json.load(f)

        # tokenizer and processer --- assume paligemma for now
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=num_image_tokens,
            max_seq_len=max_seq_len,
            tokenizer_padding=tokenizer_padding,
        )

    def reset(self):
        pass

    def preprocess(
        self,
        env,
        obs: dict,
        instruction: str,
    ) -> dict:
        """
        using sxyz convention for euler angles

        NOTE(YL): we make the env: Optional[object] to be compatible with other envs
                if env is None, obs is a dict with key "image"
        """
        if env is None:
            image = obs["image"]  # [H, W, 3]
        else:
            image = get_image_from_maniskill2_obs_dict(env, obs)  # [H, W, 3]
        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_LANCZOS4,
        )
        # no normalization for image before processor
        # always on cpu
        images = torch.as_tensor(image, dtype=torch.uint8).permute(2, 0, 1)[
            None
        ]  # [1, 3, H, W]
        model_inputs = self.processor(text=[instruction], images=images)

        # process proprio depending on the robot
        if "proprio" in obs:
            raw_proprio = obs["proprio"]
        else:
            raw_proprio = self.preprocess_proprio(obs)

        # normalize proprios - gripper opening is normalized
        if self.proprio_normalization_type == "bound":
            proprio = self.normalize_bound(
                raw_proprio,
                np.array(self.dataset_statistics["proprio"]["p01"]),
                np.array(self.dataset_statistics["proprio"]["p99"]),
                clip_min=-1,
                clip_max=1,
            )
        elif self.proprio_normalization_type == "gaussian":
            proprio = self.normalize_gaussian(
                raw_proprio,
                np.array(self.dataset_statistics["proprio"]["mean"]),
                np.array(self.dataset_statistics["proprio"]["std"]),
            )

        return {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"],
            "attention_mask": model_inputs["attention_mask"],
            "proprios": torch.as_tensor(proprio, dtype=torch.float32)[
                None, None
            ],  # [B, T, dim]
        }

    def postprocess(
        self,
        actions: np.array,
    ) -> List[dict]:
        # gripper action is not normalized in training dataset
        if self.action_normalization_type == "bound":
            raw_actions_except_gripper = self.denormalize_bound(
                actions[:, :-1],
                np.array(self.dataset_statistics["action"]["p01"])[:-1],
                np.array(self.dataset_statistics["action"]["p99"])[:-1],
                clip_min=-1,
                clip_max=1,
            )
        elif self.action_normalization_type == "gaussian":
            raw_actions_except_gripper = self.denormalize_gaussian(
                actions[:, :-1],
                np.array(self.dataset_statistics["action"]["mean"])[:-1],
                np.array(self.dataset_statistics["action"]["std"])[:-1],
            )
        raw_actions = np.concatenate(
            [
                raw_actions_except_gripper,
                actions[:, -1:],
            ],
            axis=1,
        )

        # prepare for simpler env
        actions = np.zeros((len(raw_actions), 7))  # chunk
        for idx, raw_action in enumerate(raw_actions):
            roll, pitch, yaw = raw_action[3:6]
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_gripper = self.postprocess_gripper(raw_action[-1])

            actions[idx] = np.concatenate(
                [
                    raw_action[:3],
                    action_rotation_ax * action_rotation_angle,
                    [action_gripper],
                ]
            )
        return actions

    def preprocess_proprio(self, obs: dict) -> np.array:
        raise NotImplementedError

    def postprocess_gripper(self, action: float) -> float:
        raise NotImplementedError

    def get_video_frame(self, env, obs: dict) -> np.array:
        """for recording video"""
        if env is None:
            return obs["image"]
        else:
            return get_image_from_maniskill2_obs_dict(env, obs)


class BridgeSimplerAdapter(SimplerAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203

    def reset(self):
        super().reset()

    def preprocess_proprio(self, obs: dict) -> np.array:
        # convert ee rotation to the frame of top-down
        proprio = obs["agent"]["eef_pos"]
        rm_bridge = quat2mat(proprio[3:7])
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7]
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper_openness],
            ]
        )
        return raw_proprio

    def postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 close, 1 open for simpler
        action_gripper = 2.0 * (action > 0.5) - 1.0
        return action_gripper


class EDRSimplerAdapter(SimplerAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Constants
        self.sticky_gripper_num_repeat = 15  # same used in Octo. Note this is for every individual action, not every action chunk. Control freq is 3Hz, so roughly sticky for 5 seconds.

    def reset(self):
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.previous_gripper_action = None
        super().reset()

    def preprocess_proprio(self, obs: dict) -> np.array:
        """convert wxyz quat from simpler to xyzw used in fractal"""
        quat_xyzw = np.roll(obs["agent"]["eef_pos"][3:7], -1)
        gripper_width = obs["agent"]["eef_pos"][
            7
        ]  # from simpler, 0 for close, 1 for open
        gripper_closedness = (
            1 - gripper_width
        )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
        raw_proprio = np.concatenate(
            (
                obs["agent"]["eef_pos"][:3],
                quat_xyzw,
                [gripper_closedness],
            )
        )
        return raw_proprio

    def postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L187"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 open, 1 close for simpler

        action = (action * 2) - 1  # [0, 1] -> [-1, 1] -1 close, 1 open

        # without sticky
        relative_gripper_action = -action
        # if self.previous_gripper_action is None:
        #     relative_gripper_action = -1  # open
        # else:
        #     relative_gripper_action = -action
        # self.previous_gripper_action = action

        # switch to sticky closing
        if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative_gripper_action

        # sticky closing
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative_gripper_action = self.sticky_gripper_action

        # reaching maximum sticky
        if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0

        return relative_gripper_action
