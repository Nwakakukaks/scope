"""Distortion effects for Cosmic VFX."""
import math

import torch
import torch.nn.functional as F


def apply_distortion(
    frames: torch.Tensor,
    intensity: float,
    scale: float,
    time: float,
    variant: str,
    device: torch.device,
) -> torch.Tensor:
    """Apply distortion effects (wave, pinch, barrel) to frames."""
    result = frames.clone()
    h, w = frames.shape[1], frames.shape[2]

    if variant == "pinch":
        # Animated pinch — strength pulses with time
        pulse = 0.7 + 0.3 * math.sin(time * 2.0)
        strength = intensity * 0.5 * pulse
        # Use grid_sample for performance
        grid_y = torch.linspace(-1, 1, h, device=device)
        grid_x = torch.linspace(-1, 1, w, device=device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        dist = torch.sqrt(gy**2 + gx**2)
        factor = 1.0 - strength * (1.0 - dist.clamp(0, 1))
        new_gy = gy * factor
        new_gx = gx * factor
        grid = torch.stack([new_gx, new_gy], dim=-1).unsqueeze(0)
        grid = grid.expand(frames.shape[0], -1, -1, -1)
        result = F.grid_sample(
            frames.permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).permute(0, 2, 3, 1)

    elif variant == "barrel":
        # Animated barrel — strength pulses
        pulse = 0.7 + 0.3 * math.sin(time * 1.5)
        strength = intensity * 0.5 * pulse
        grid_y = torch.linspace(-1, 1, h, device=device)
        grid_x = torch.linspace(-1, 1, w, device=device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        dist = torch.sqrt(gy**2 + gx**2)
        factor = 1.0 + strength * (1.0 - dist.clamp(0, 1))
        new_gy = gy * factor
        new_gx = gx * factor
        grid = torch.stack([new_gx, new_gy], dim=-1).unsqueeze(0)
        grid = grid.expand(frames.shape[0], -1, -1, -1)
        result = F.grid_sample(
            frames.permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).permute(0, 2, 3, 1)

    else:  # wave
        # Animated wave — vectorized via grid_sample
        phase = time * 3.0
        grid_y = torch.linspace(-1, 1, h, device=device)
        grid_x = torch.linspace(-1, 1, w, device=device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        # Sine wave displacement on x-axis, driven by y position
        y_norm = (gy + 1) / 2  # [0, 1]
        wave_offset = (
            torch.sin(y_norm * 6.28318 * scale + phase) * intensity * 0.1 * scale
        )
        new_gx = gx + wave_offset
        grid = torch.stack([new_gx, gy], dim=-1).unsqueeze(0)
        grid = grid.expand(frames.shape[0], -1, -1, -1)
        result = F.grid_sample(
            frames.permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).permute(0, 2, 3, 1)

    return result
