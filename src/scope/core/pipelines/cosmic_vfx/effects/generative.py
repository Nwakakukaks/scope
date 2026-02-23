"""Generative pattern effects for Cosmic VFX."""

import torch


def apply_generative(
    frames: torch.Tensor,
    intensity: float,
    time: float,
    scale: float,
    variant: str,
    device: torch.device,
) -> torch.Tensor:
    """Apply generative patterns (noise, pattern, fractal) to frames."""
    h, w = frames.shape[1], frames.shape[2]
    y_coords = torch.arange(h, dtype=torch.float32, device=device) / h
    x_coords = torch.arange(w, dtype=torch.float32, device=device) / w

    if variant == "pattern":
        # Animated sine grid — scrolls with time
        pattern = torch.sin(x_coords * 6.28 * scale * 2 + time * 2.0) * torch.sin(
            y_coords.unsqueeze(1) * 6.28 * scale * 2 + time * 1.5
        )
        pattern = (pattern + 1) / 2
        result = (
            frames * (1 - intensity * 0.5)
            + pattern.unsqueeze(0).unsqueeze(-1) * intensity * 0.5
        )

    elif variant == "fractal":
        # Animated layered sine waves
        pattern = torch.zeros(h, w, device=device)
        for octave in range(1, 5):
            freq = scale * (2**octave)
            amp = 1.0 / octave
            phase = time * (0.5 + octave * 0.3)
            pattern += (
                amp
                * torch.sin(x_coords * freq + phase)
                * torch.sin(y_coords.unsqueeze(1) * freq + phase * 0.7)
            )
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)
        result = (
            frames * (1 - intensity * 0.5)
            + pattern.unsqueeze(0).unsqueeze(-1) * intensity * 0.5
        )

    else:  # noise
        # Animated noise — changes every frame
        noise = torch.randn(h, w, device=device) * 0.5 + 0.5
        result = (
            frames * (1 - intensity * 0.3)
            + noise.unsqueeze(0).unsqueeze(-1) * intensity * 0.3
        )

    return result.clamp(0, 1)
