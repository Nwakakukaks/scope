"""Atmospheric effects for Cosmic VFX."""
import math

import torch
import torch.nn.functional as F


def apply_atmospheric(
    frames: torch.Tensor,
    intensity: float,
    time: float,
    variant: str,
    bloom_kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply atmospheric effects (fog, glow, bloom) to frames."""
    result = frames.clone()
    device = frames.device

    if variant == "glow":
        # Pulsing glow — highlight brightness oscillates
        pulse = 0.7 + 0.3 * math.sin(time * 3.0)
        glow = torch.clamp(result - 0.5, 0, 1) * 2
        result = result + glow * intensity * 0.5 * pulse

    elif variant == "bloom":
        # Animated bloom — threshold oscillates, creating breathing effect
        thresh = 0.65 + 0.05 * math.sin(time * 2.0)
        bright = torch.clamp(result - thresh, 0, 1) * 3.0
        # Fast blur of bright areas via cached conv kernel
        x = bright.permute(0, 3, 1, 2)
        if bloom_kernel is None:
            bloom_k = torch.ones(1, 1, 7, 7, device=device) / 49.0
            bloom_k = bloom_k.expand(x.shape[1], -1, -1, -1)
        else:
            bloom_k = bloom_kernel.expand(x.shape[1], -1, -1, -1)
        x = F.conv2d(x, bloom_k, padding=3, groups=x.shape[1])
        bright = x.permute(0, 2, 3, 1)
        result = result + bright * intensity * 0.5

    else:  # fog
        # Animated fog — density drifts with time
        h, w = frames.shape[1], frames.shape[2]
        y_coords = torch.arange(h, dtype=torch.float32, device=device) / h
        x_coords = torch.arange(w, dtype=torch.float32, device=device) / w
        # Fog density varies spatially and temporally
        fog_pattern = torch.sin(x_coords * 3.0 + time * 0.5) * torch.sin(
            y_coords.unsqueeze(1) * 2.0 + time * 0.3
        )
        fog_pattern = (fog_pattern + 1) / 2  # [0, 1]
        fog = fog_pattern.unsqueeze(0).unsqueeze(-1) * 0.5  # fog color ~gray
        fog_amount = intensity * 0.3
        result = (
            result * (1 - fog_amount)
            + fog * fog_amount
            + 0.15 * intensity * fog_pattern.unsqueeze(0).unsqueeze(-1)
        )

    return result.clamp(0, 1)
