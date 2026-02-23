"""Halftone (newspaper dot pattern) effect – pure PyTorch, GPU-friendly."""

import torch
import torch.nn.functional as F


def halftone(
    frames: torch.Tensor,
    dot_size: int = 8,
    sharpness: float = 0.7,
) -> torch.Tensor:
    """Apply a halftone dot-pattern effect to video frames.

    Dark regions produce large dots; bright regions produce small dots,
    mimicking classic newspaper / pop-art printing.

    Args:
        frames: (T, H, W, C) float32 tensor in [0, 1].
        dot_size: Diameter of each halftone cell in pixels (4–20).
        sharpness: Edge sharpness of the dots (0 = very soft, 1 = hard).

    Returns:
        Tensor of same shape with halftone pattern applied.
    """
    if dot_size < 2:
        return frames

    T, H, W, C = frames.shape
    device = frames.device
    cell = int(dot_size)

    # --- 1. Luminance for dot sizing ---
    luma = 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]

    # --- 2. Cell-averaged luminance via avg_pool2d ---
    pad_h = (cell - H % cell) % cell
    pad_w = (cell - W % cell) % cell
    luma_4d = luma.unsqueeze(1)  # (T, 1, H, W)
    luma_padded = F.pad(luma_4d, (0, pad_w, 0, pad_h), mode="reflect")
    cell_luma = F.avg_pool2d(luma_padded, cell, cell)
    cell_luma = F.interpolate(
        cell_luma, size=(H + pad_h, W + pad_w), mode="nearest"
    )
    cell_luma = cell_luma[:, 0, :H, :W]  # (T, H, W)

    # --- 3. Distance from each pixel to its cell centre ---
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(y, x, indexing="ij")

    local_y = (gy % cell) - (cell - 1) / 2.0
    local_x = (gx % cell) - (cell - 1) / 2.0
    dist = torch.sqrt(local_x * local_x + local_y * local_y)  # (H, W)

    # --- 4. Dot radius from brightness (dark → big, bright → small) ---
    max_r = cell / 2.0
    dot_r = max_r * torch.sqrt((1.0 - cell_luma).clamp(0, 1))  # (T, H, W)

    # --- 5. Soft-edge mask via sigmoid ---
    edge = max(0.3, (1.0 - sharpness) * max_r)
    mask = torch.sigmoid((dot_r - dist.unsqueeze(0)) / edge * 6.0)  # (T, H, W)

    # --- 6. Composite: original colour in dots, white background ---
    mask = mask.unsqueeze(-1)  # (T, H, W, 1)
    result = frames * mask + (1.0 - mask)

    return result
