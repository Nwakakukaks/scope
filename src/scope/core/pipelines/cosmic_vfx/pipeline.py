"""Cosmic VFX Pipeline."""

from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .effects import (
    apply_atmospheric,
    apply_blend,
    apply_blur,
    apply_color,
    apply_distortion,
    apply_edge,
    apply_generative,
    apply_glitch,
    apply_retro,
    apply_utility,
)
from .schema import CosmicVFXConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

# Processing order for stacked effects
_CATEGORY_ORDER = [
    "glitch",
    "retro",
    "distortion",
    "color",
    "blend",
    "edge",
    "blur",
    "generative",
    "atmospheric",
    "utility",
]


class CosmicVFXPipeline(Pipeline):
    """Cosmic VFX Plugin — real-time visual effects with stackable shaders.

    Supports 10 categories with 3 variants each (30 shaders total).
    Multiple categories can be enabled simultaneously for MOSH Pro-style stacking.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return CosmicVFXConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.frame_count = 0

        # Pre-allocate constant kernels (avoids per-frame GPU allocation)
        self._sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 3)
        self._sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 3)
        self._bloom_kernel = torch.ones(1, 1, 7, 7, device=self.device) / 49.0

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Cosmic VFX pipeline requires video input")

        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # Debug: Log pipeline execution
        print(f"[CosmicVFX] Processing {frames.shape} frames on {self.device}")

        # Global (master) parameters
        master_intensity = float(kwargs.get("intensity", 1.0))
        speed = float(kwargs.get("speed", 1.0))
        scale = float(kwargs.get("scale", 1.0))
        hue_shift = float(kwargs.get("hue_shift", 0.0))
        saturation = float(kwargs.get("saturation", 1.0))
        brightness = float(kwargs.get("brightness", 1.0))
        blend_mode = str(kwargs.get("blend_mode", "normal"))

        time = self.frame_count * speed * 0.016

        result = frames
        active_effects = []

        # Stack enabled categories in order
        for cat in _CATEGORY_ORDER:
            enabled = kwargs.get(f"enable_{cat}", False)
            if not enabled:
                continue
            variant = str(kwargs.get(f"{cat}_shader", ""))
            cat_intensity = (
                float(kwargs.get(f"{cat}_intensity", 1.0)) * master_intensity
            )

            active_effects.append(f"{cat}({variant}, {cat_intensity:.2f})")

            try:
                if cat == "glitch":
                    result = apply_glitch(
                        result, cat_intensity, time, variant, self.device
                    )
                elif cat == "retro":
                    result = apply_retro(
                        result, cat_intensity, time, variant, self.device
                    )
                elif cat == "distortion":
                    result = apply_distortion(
                        result, cat_intensity, scale, time, variant, self.device
                    )
                elif cat == "color":
                    result = apply_color(
                        result, cat_intensity, hue_shift, saturation, variant
                    )
                elif cat == "blend":
                    result = apply_blend(result, cat_intensity, variant)
                elif cat == "edge":
                    result = apply_edge(
                        result, cat_intensity, variant, self._sobel_x, self._sobel_y
                    )
                elif cat == "blur":
                    result = apply_blur(result, cat_intensity, scale, variant)
                elif cat == "generative":
                    result = apply_generative(
                        result, cat_intensity, time, scale, variant, self.device
                    )
                elif cat == "atmospheric":
                    result = apply_atmospheric(
                        result, cat_intensity, time, variant, self._bloom_kernel
                    )
                elif cat == "utility":
                    result = apply_utility(result, cat_intensity, variant)
            except Exception as e:
                print(f"[CosmicVFX ERROR] {cat} effect failed: {e}")
                import traceback

                traceback.print_exc()

        if active_effects:
            print(f"[CosmicVFX] Applied effects: {', '.join(active_effects)}")

        # Global blend mode compositing
        if blend_mode != "normal":
            result = self._composite(frames, result, blend_mode, master_intensity)

        result = result * brightness

        self.frame_count = (self.frame_count + 1) % 100000
        return {"video": result.clamp(0, 1)}

    @staticmethod
    def _composite(
        original: torch.Tensor, effect: torch.Tensor, mode: str, intensity: float
    ) -> torch.Tensor:
        mix = intensity * 0.5
        if mode == "screen":
            blended = 1.0 - (1.0 - original) * (1.0 - effect)
        elif mode == "multiply":
            blended = original * effect
        elif mode == "overlay":
            blended = torch.where(
                original < 0.5,
                2 * original * effect,
                1.0 - 2 * (1.0 - original) * (1.0 - effect),
            )
        else:
            return effect
        return original * (1.0 - mix) + blended * mix
