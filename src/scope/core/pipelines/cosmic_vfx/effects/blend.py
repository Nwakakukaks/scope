"""Blend effects for Cosmic VFX."""
import torch


def apply_blend(
    frames: torch.Tensor,
    intensity: float,
    variant: str,
) -> torch.Tensor:
    """Apply blend modes (screen, multiply, overlay) to frames."""
    result = frames.clone()
    mix = intensity * 0.5

    if variant == "screen":
        result = 1.0 - (1.0 - result) * (1.0 - mix)
    elif variant == "multiply":
        result = result * (1.0 - mix)
    elif variant == "overlay":
        result = torch.where(
            result < 0.5,
            2 * result * (1.0 - mix),
            1.0 - 2 * (1.0 - result) * (1.0 - mix),
        )
    return result.clamp(0, 1)
