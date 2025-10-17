# agents/dave2_agent.py
import carla
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .models import DAVE2v1


class DAVE2Agent:
    """
    Minimal inference wrapper that predicts steering from an RGB frame.
    Pair it with your own longitudinal controller (e.g., PID on speed).
    """

    def __init__(
        self, ego: carla.Vehicle, ckpt_path: str, input_shape=(180, 320), device="cuda"
    ):
        self.ego = ego
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = DAVE2v1(input_shape=input_shape).to(self.device)
        # state = torch.load(ckpt_path, map_location=self.device)
        # self.model.load_state_dict(state["model"])

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Handle either {"model": ...} / {"state_dict": ...} or a bare state dict
        sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt

        # Remove non-param stats saved with some checkpoints
        for k in ("speed_mean", "speed_std"):
            if k in sd:
                print(f"[WARN] dropping non-param key from state_dict: {k}")
                sd.pop(k)

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected:
            print("[WARN] Unexpected keys ignored:", unexpected)
        if missing:
            print("[WARN] Missing keys (not in checkpoint):", missing)

        self.model.eval()
        self.tf = Compose(
            [
                Resize(input_shape),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @torch.no_grad()
    def predict_steer(self, rgb_pil):
        x = self.tf(rgb_pil).unsqueeze(0).to(self.device)
        steer = self.model(x).item()  # [-1,1]
        return float(steer)

    def run_step(self, rgb_pil, throttle=0.3, brake=0.0):
        steer = self.predict_steer(rgb_pil)
        ctrl = carla.VehicleControl()
        ctrl.throttle = max(0.0, min(1.0, throttle))
        ctrl.brake = max(0.0, min(1.0, brake))
        ctrl.steer = max(-1.0, min(1.0, steer))
        ctrl.hand_brake = False
        ctrl.reverse = False
        return ctrl
