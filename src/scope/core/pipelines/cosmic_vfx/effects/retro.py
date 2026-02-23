"""Retro effects for Cosmic VFX."""
import math

import torch
import torch.nn.functional as F


def apply_retro(
    frames: torch.Tensor,
    intensity: float,
    time: float,
    variant: str,
    device: torch.device,
) -> torch.Tensor:
    """Apply retro/VHS/CRT effects to frames."""
    result = frames.clone()
    h, w = frames.shape[1], frames.shape[2]

    if variant == "crt":
        # Rolling CRT scanlines — vectorized mask
        scan_spacing = max(2, int(4 / (intensity + 0.1)))
        roll_offset = int(time * 60) % scan_spacing

        # Vectorized scanline mask (no loop)
        y_indices = torch.arange(h, device=device)
        is_scanline = ((y_indices - roll_offset) % scan_spacing) == 0
        mask = torch.where(is_scanline, 1.0 - 0.5 * intensity, 1.0)
        result = result * mask.view(1, h, 1, 1)

        # Slight horizontal jitter (CRT wobble)
        jitter = int(math.sin(time * 8.0) * intensity * 3)
        if jitter != 0:
            result = torch.roll(result, jitter, dims=2)

        # Green phosphor tint
        result[:, :, :, 1] *= 1.0 + 0.08 * intensity
        # Slight vignette darkening at edges
        result = result.clamp(0, 1)

    elif variant == "pixelate":
        # Animated pixelation — block size pulses with time
        pulse = 0.8 + 0.2 * math.sin(time * 2.0)
        block = max(2, int(intensity * 16 * pulse))
        bh, bw = max(1, h // block), max(1, w // block)
        downsampled = F.interpolate(
            result.permute(0, 3, 1, 2), size=(bh, bw), mode="area"
        )
        result = F.interpolate(downsampled, size=(h, w), mode="nearest").permute(
            0, 2, 3, 1
        )

    else:  # vhs
        # VHS — vectorized scanlines + animated noise + color banding
        scan_spacing = max(1, int(3 * intensity))
        roll_offset = int(time * 30) % max(1, scan_spacing)
        y_indices = torch.arange(h, device=device)
        is_scanline = ((y_indices - roll_offset) % scan_spacing) == 0
        vhs_mask = torch.where(is_scanline, 1.0 - 0.3 * intensity, 1.0)
        result = result * vhs_mask.view(1, h, 1, 1)

        # Animated noise (changes every frame)
        noise = torch.randn_like(result) * intensity * 0.1
        result = (result + noise).clamp(0, 1)

        # VHS tracking bar — a bright/dark band that rolls down
        bar_pos = int(time * 20) % h
        bar_height = max(2, int(h * 0.03))
        bar_end = min(bar_pos + bar_height, h)
        result[:, bar_pos:bar_end, :, :] *= 1.0 + 0.4 * intensity

        # Color banding
        if intensity > 0.3:
            levels = max(2, int(8 / intensity))
            result = (result * levels).round() / levels

    return result.clamp(0, 1)
