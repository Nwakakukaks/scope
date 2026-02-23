"""Color effects for Cosmic VFX."""
import math

import torch


def apply_color(
    frames: torch.Tensor,
    intensity: float,
    hue_shift: float,
    saturation: float,
    variant: str,
) -> torch.Tensor:
    """Apply color adjustments (hue shift, saturation, color grade) to frames."""
    result = frames.clone()
    r = frames[:, :, :, 0]
    g = frames[:, :, :, 1]
    b = frames[:, :, :, 2]

    if variant == "saturate":
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        sat = saturation * intensity
        result[:, :, :, 0] = gray + (r - gray) * sat
        result[:, :, :, 1] = gray + (g - gray) * sat
        result[:, :, :, 2] = gray + (b - gray) * sat

    elif variant == "grade":
        shadow_tint = intensity * 0.15
        highlight_tint = intensity * 0.1
        result[:, :, :, 0] = r + shadow_tint * (1.0 - r)
        result[:, :, :, 1] = g  # green channel untouched
        result[:, :, :, 2] = b + highlight_tint * r

    else:  # hueshift
        # Proper hue rotation using rotation matrix
        angle = hue_shift * intensity * 3.14159
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # Rotation in YIQ-like space
        result[:, :, :, 0] = (
            r * (0.299 + 0.701 * cos_a + 0.168 * sin_a)
            + g * (0.587 - 0.587 * cos_a + 0.330 * sin_a)
            + b * (0.114 - 0.114 * cos_a - 0.497 * sin_a)
        )
        result[:, :, :, 1] = (
            r * (0.299 - 0.299 * cos_a - 0.328 * sin_a)
            + g * (0.587 + 0.413 * cos_a + 0.035 * sin_a)
            + b * (0.114 - 0.114 * cos_a + 0.292 * sin_a)
        )
        result[:, :, :, 2] = (
            r * (0.299 - 0.299 * cos_a + 1.250 * sin_a)
            + g * (0.587 - 0.588 * cos_a - 1.050 * sin_a)
            + b * (0.114 + 0.886 * cos_a - 0.203 * sin_a)
        )
        # Apply saturation on top
        if abs(saturation - 1.0) > 0.01:
            gray = (
                0.299 * result[:, :, :, 0]
                + 0.587 * result[:, :, :, 1]
                + 0.114 * result[:, :, :, 2]
            ).unsqueeze(-1)
            result = gray + (result - gray) * saturation

    return result.clamp(0, 1)
