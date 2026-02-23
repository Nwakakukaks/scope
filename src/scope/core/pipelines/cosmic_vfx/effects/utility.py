"""Utility effects for Cosmic VFX."""
import torch


def apply_utility(
    frames: torch.Tensor,
    intensity: float,
    variant: str,
) -> torch.Tensor:
    """Apply utility effects (invert, posterize, threshold) to frames."""
    if variant == "posterize":
        levels = max(2, int(256 / (1 + intensity * 10)))
        result = (frames * levels).round() / levels

    elif variant == "threshold":
        gray = (
            0.299 * frames[:, :, :, 0]
            + 0.587 * frames[:, :, :, 1]
            + 0.114 * frames[:, :, :, 2]
        )
        thresh = 0.5 * (1.0 - intensity * 0.5)
        binary = (gray > thresh).float()
        result = binary.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

    else:  # invert
        result = 1.0 - frames

    return result.clamp(0, 1)
