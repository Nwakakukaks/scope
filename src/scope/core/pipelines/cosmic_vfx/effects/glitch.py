"""Glitch effects for Cosmic VFX."""
import math

import torch
import torch.nn.functional as F


def apply_glitch(
    frames: torch.Tensor,
    intensity: float,
    time: float,
    variant: str,
    device: torch.device,
) -> torch.Tensor:
    """Apply glitch effects to frames."""
    result = frames.clone()
    h = frames.shape[1]

    if variant == "rgb-split":
        # Animated RGB channel separation — shift oscillates with time
        base_shift = int(intensity * 15)
        anim = math.sin(time * 3.0)
        shift_r = int(base_shift * (0.5 + 0.5 * anim))
        shift_b = int(base_shift * (0.5 - 0.5 * anim))
        result[:, :, :, 0] = torch.roll(result[:, :, :, 0], shift_r, dims=2)
        result[:, :, :, 2] = torch.roll(result[:, :, :, 2], -shift_b, dims=2)

    elif variant == "scanline":
        # Animated scanline displacement — vectorized band shifts
        band_height = max(2, int(8 / (intensity + 0.1)))
        offset = int(time * 40 * intensity) % h
        y_indices = torch.arange(h, device=device)
        # Determine which rows belong to displaced bands
        shifted_y = (y_indices - offset) % h
        in_band = (shifted_y % (band_height * 2)) < band_height
        # Compute per-row shift amounts
        shifts = (
            intensity
            * 30
            * torch.sin(
                torch.tensor(time * 2.0, device=device)
                + y_indices.float() * 0.1
            )
        ).long()
        shifts = shifts * in_band.long()
        # Apply shifts using grid_sample for vectorized displacement
        w = result.shape[2]
        grid_x = torch.arange(w, device=device).float()
        # Build per-row x-coordinates with shifts applied
        new_x = (grid_x.unsqueeze(0) - shifts.unsqueeze(1).float()) % w
        new_x = new_x / (w - 1) * 2 - 1  # normalize to [-1, 1]
        grid_y = y_indices.float() / (h - 1) * 2 - 1
        grid = (
            torch.stack(
                [
                    new_x,
                    grid_y.unsqueeze(1).expand(-1, w),
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .expand(result.shape[0], -1, -1, -1)
        )
        result = F.grid_sample(
            result.permute(0, 3, 1, 2),
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        ).permute(0, 2, 3, 1)

    else:  # basic
        # Random glitch lines — different each frame due to randomness
        num_glitches = max(1, int(intensity * 8))
        for _ in range(num_glitches):
            y = torch.randint(0, h, (1,)).item()
            band = max(1, torch.randint(1, max(2, int(intensity * 6)), (1,)).item())
            shift = int((torch.rand(1).item() - 0.5) * intensity * 40)
            y_end = min(y + band, h)
            result[:, y:y_end, :, :] = torch.roll(
                result[:, y:y_end, :, :], shift, dims=2
            )
        # Animated color channel offset
        if intensity > 0.3:
            shift_amount = int(intensity * 10 * math.sin(time * 5.0))
            result[:, :, :, 0] = torch.roll(
                result[:, :, :, 0], shift_amount, dims=2
            )

    return result
