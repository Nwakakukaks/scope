"""Blur effects for Cosmic VFX."""
import torch
import torch.nn.functional as F


def apply_blur(
    frames: torch.Tensor,
    intensity: float,
    scale: float,
    variant: str,
) -> torch.Tensor:
    """Apply blur effects (gaussian, motion, radial) to frames."""
    passes = max(1, int(intensity * scale * 5))
    # Work in NCHW for conv operations
    x = frames.permute(0, 3, 1, 2)  # (T, C, H, W)

    if variant == "motion":
        # Horizontal motion blur via averaging kernel
        k = min(passes * 2 + 1, 31)
        if k >= 3:
            kernel = torch.ones(1, 1, 1, k, device=frames.device) / k
            kernel = kernel.expand(x.shape[1], -1, -1, -1)
            pad = k // 2
            x = F.conv2d(x, kernel, padding=(0, pad), groups=x.shape[1])

    elif variant == "radial":
        # Accumulate shifted copies and average once at the end
        num_shifts = min(passes, 7)
        accumulated = x.clone()
        for i in range(1, num_shifts + 1):
            accumulated = (
                accumulated + torch.roll(x, i, dims=2) + torch.roll(x, -i, dims=2)
            )
            accumulated = (
                accumulated + torch.roll(x, i, dims=3) + torch.roll(x, -i, dims=3)
            )
        x = accumulated / (1 + num_shifts * 4)

    else:  # gaussian
        # Box blur approximation (2 passes for smoother result)
        k = min(passes * 2 + 1, 31)
        if k >= 3:
            kernel_h = torch.ones(1, 1, k, 1, device=frames.device) / k
            kernel_w = torch.ones(1, 1, 1, k, device=frames.device) / k
            kernel_h = kernel_h.expand(x.shape[1], -1, -1, -1)
            kernel_w = kernel_w.expand(x.shape[1], -1, -1, -1)
            pad = k // 2
            x = F.conv2d(x, kernel_h, padding=(pad, 0), groups=x.shape[1])
            x = F.conv2d(x, kernel_w, padding=(0, pad), groups=x.shape[1])

    return x.permute(0, 2, 3, 1)
