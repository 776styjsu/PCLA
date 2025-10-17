# models/dave2.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DAVE2v1"]


class DAVE2v1(nn.Module):
    """
    Standard DAVE-2 style net:
      5 conv layers -> 3 FC -> tanh steering in [-1, 1]
    """

    def __init__(self, input_shape=(180, 320)):
        super().__init__()
        self.input_shape = input_shape  # (H, W)

        self.bn1 = nn.BatchNorm2d(3, eps=1e-3, momentum=0.99, track_running_stats=False)
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.input_shape)
            feat = self.conv5(
                F.relu(
                    self.conv4(
                        F.relu(
                            self.conv3(
                                F.relu(self.conv2(F.relu(self.conv1(self.bn1(dummy)))))
                            )
                        )
                    )
                )
            )
            flat_dim = int(np.prod(feat.shape[1:]))

        self.fc1 = nn.Linear(flat_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.out = nn.Linear(10, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.out(x))  # steering in [-1, 1]
        return x
