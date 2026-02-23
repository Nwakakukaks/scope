"""Edge detection effects for Cosmic VFX."""
import torch
import torch.nn.functional as F


def apply_edge(
    frames: torch.Tensor,
    intensity: float,
    variant: str,
    sobel_x: torch.Tensor | None = None,
    sobel_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply edge detection effects (sobel, outline, neon) to frames."""
    h, w = frames.shape[1], frames.shape[2]
    device = frames.device

    # Create Sobel kernels if not provided
    if sobel_x is None:
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)
    if sobel_y is None:
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)

    # Sobel via conv2d with cached kernels
    gray = (
        0.299 * frames[:, :, :, 0]
        + 0.587 * frames[:, :, :, 1]
        + 0.114 * frames[:, :, :, 2]
    ).unsqueeze(1)  # (T, 1, H, W)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    edges = torch.sqrt(gx**2 + gy**2).squeeze(1)  # (T, H, W)
    edges = edges / (edges.max() + 1e-6)

    if variant == "outline":
        # Hard binary edges — clean white lines on black
        threshold = 0.15 / (intensity + 0.1)
        binary = (edges > threshold).float()
        result = binary.unsqueeze(-1).expand(-1, -1, -1, 3) * intensity

    elif variant == "neon":
        edge_rgb = edges.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        # Neon cyan/magenta tint
        neon = edge_rgb.clone()
        neon[:, :, :, 0] *= 0.8
        neon[:, :, :, 1] *= 1.3
        neon[:, :, :, 2] *= 1.5
        result = frames * (1.0 - intensity * 0.5) + neon * intensity

    else:  # sobel
        # Gradient magnitude edges — smooth grayscale
        result = edges.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous() * intensity

    return result.clamp(0, 1)
