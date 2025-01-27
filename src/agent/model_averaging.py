import logging

import torch

log = logging.getLogger(__name__)


class ModelAveraging:
    """Not supporting resume from checkpoint currently"""

    def __init__(self, model, cfg, device):
        self.use_ema = cfg.get("use_ema", False)
        self.use_swa = cfg.get("use_swa", False)
        assert not (self.use_ema and self.use_swa), (
            "Cannot use both EMA and SWA at once"
        )

        self.model_avg = None
        self.model = model
        self.device = device

        # EMA configuration
        if self.use_ema:
            self.ema_start = cfg.ema_start
            self.ema_decay = cfg.get("ema_decay", 0.99)
            self.ema_freq = cfg.get("ema_freq", 1)
            self.ema_device = cfg.get("ema_device", self.device)

        # SWA configuration
        if self.use_swa:
            self.swa_start = cfg.swa_start
            self.swa_freq = cfg.swa_freq
            self.swa_device = cfg.get("swa_device", "cpu")

    def maybe_initialize(self, cnt_update):
        if self.use_swa and cnt_update == self.swa_start:
            self.model_avg = torch.optim.swa_utils.AveragedModel(
                self.model, device=self.swa_device
            )
            logging.info("Starting SWA...")

        if self.use_ema and cnt_update == self.ema_start:
            self.model_avg = torch.optim.swa_utils.AveragedModel(
                self.model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.ema_decay),
                device=self.ema_device,
            )
            logging.info(f"Starting EMA with decay {self.ema_decay}...")

    def maybe_update(self, cnt_update):
        if self.model_avg is None:
            return
        if self.use_ema and cnt_update % self.ema_freq == 0:
            self.model_avg.update_parameters(self.model)
            logging.info("EMA updated")
        if self.use_swa and cnt_update % self.swa_freq == 0:
            self.model_avg.update_parameters(self.model)
            logging.info("SWA updated")

    def get_model_module(self) -> dict:
        if self.model_avg:
            return self.model_avg.module
        return self.model

    def state_dict(self) -> dict:
        if self.model_avg:
            return {
                "state_dict": self.model_avg.module.state_dict(),
                "n_averaged": self.model_avg.state_dict().get("n_averaged", 1),
                "model_type": "ema" if self.use_ema else "swa",
            }
        return {}
