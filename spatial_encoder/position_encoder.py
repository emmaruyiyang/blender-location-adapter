"""
Position encoder MLP: 6D spatial feature -> VLM embedding space
"""

import torch
import torch.nn as nn


class PositionEncoder(nn.Module):
    """
    Maps a 6-dim spatial feature vector to the VLM's hidden dimension.

    input:  [x, y, z, distance, azimuth, elevation]  (z-score normalized)
    output: (hidden_size,) embedding, same dtype as VLM token embeddings
    """

    def __init__(self, hidden_size: int = 3584, input_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_objects, 6) or (num_objects, 6)
        returns: same shape but last dim = hidden_size
        """
        return self.net(x)


def get_position_encoder(model_name_or_hidden_size) -> PositionEncoder:
    """Convenience: build encoder matched to a known model."""
    hidden_sizes = {
        "Qwen/Qwen2-VL-7B-Instruct": 3584,
        "Qwen/Qwen2-VL-2B-Instruct": 1536,
        "llava-hf/llava-1.5-7b-hf":  4096,
    }
    if isinstance(model_name_or_hidden_size, int):
        hidden_size = model_name_or_hidden_size
    else:
        hidden_size = hidden_sizes.get(model_name_or_hidden_size, 3584)
    return PositionEncoder(hidden_size=hidden_size)
