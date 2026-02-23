from typing import Literal

from pydantic import Field
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)

# ── Per-Category Shader Types (each dropdown is locked to its category) ──
GlitchShader = Literal["basic", "rgb-split", "scanline"]
RetroShader = Literal["vhs", "crt", "pixelate"]
DistortionShader = Literal["wave", "pinch", "barrel"]
ColorShader = Literal["hueshift", "saturate", "grade"]
BlendShader = Literal["screen", "multiply", "overlay"]
EdgeShader = Literal["sobel", "outline", "neon"]
BlurShader = Literal["gaussian", "motion", "radial"]
GenerativeShader = Literal["noise", "pattern", "fractal"]
AtmosphericShader = Literal["fog", "glow", "bloom"]
UtilityShader = Literal["invert", "posterize", "threshold"]

# ── Blend Modes ────────────────────────────────────────────────────
BlendMode = Literal["normal", "screen", "multiply", "overlay"]


class CosmicVFXConfig(BasePipelineConfig):
    """Configuration for the Cosmic VFX Plugin with 30 shaders across 10 categories."""

    pipeline_id = "cosmic-vfx"
    pipeline_name = "Cosmic VFX"
    pipeline_description = (
        "Real-time visual effects: Glitch, Retro, Distortion, "
        "Color, Blend, Edge, Blur, Generative, Atmospheric, and Utility"
    )

    supports_prompts = False

    modes = {"video": ModeDefaults(default=True)}

    usage = [UsageType.POSTPROCESSOR]

    # ── Per-category enable toggles + shader selectors ──────────
    # Toggle ON to activate a category. Stack multiple for MOSH Pro-style chaining.

    enable_glitch: bool = Field(
        default=False,
        description="Digital artifacts and data corruption effects",
        json_schema_extra=ui_field_config(
            order=10,
            label="Enable Glitch",
        ),
    )
    glitch_shader: GlitchShader = Field(
        default="basic",
        description="basic = random color bands, rgb-split = channel separation, scanline = horizontal band displacement",
        json_schema_extra=ui_field_config(
            order=11,
            label="Glitch Shader",
        ),
    )

    glitch_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Per-category intensity for Glitch. Multiplied by global Intensity.",
        json_schema_extra=ui_field_config(
            order=12,
            label="Glitch Intensity",
        ),
    )

    enable_retro: bool = Field(
        default=False,
        description="Vintage video and old-school display looks",
        json_schema_extra=ui_field_config(
            order=13,
            label="Enable Retro",
        ),
    )
    retro_shader: RetroShader = Field(
        default="vhs",
        description="vhs = tape noise and tracking bars, crt = rolling scanlines, pixelate = mosaic blocks",
        json_schema_extra=ui_field_config(
            order=14,
            label="Retro Shader",
        ),
    )

    retro_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Per-category intensity for Retro. Multiplied by global Intensity.",
        json_schema_extra=ui_field_config(
            order=15,
            label="Retro Intensity",
        ),
    )

    enable_distortion: bool = Field(
        default=False,
        description="Warp and bend the image geometry",
        json_schema_extra=ui_field_config(
            order=16,
            label="Enable Distortion",
        ),
    )
    distortion_shader: DistortionShader = Field(
        default="wave",
        description="wave = animated sine ripple, pinch = center squeeze, barrel = fisheye lens",
        json_schema_extra=ui_field_config(
            order=18,
            label="Distortion Shader",
        ),
    )

    distortion_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Per-category intensity for Distortion. Multiplied by global Intensity.",
        json_schema_extra=ui_field_config(
            order=19,
            label="Distortion Intensity",
        ),
    )

    enable_color: bool = Field(
        default=False,
        description="Shift hues, adjust saturation, and grade tones",
        json_schema_extra=ui_field_config(
            order=20,
            label="Enable Color",
        ),
    )
    color_shader: ColorShader = Field(
        default="hueshift",
        description="hueshift = rotate color wheel, saturate = boost or drain color, grade = shadow/highlight toning",
        json_schema_extra=ui_field_config(
            order=22,
            label="Color Shader",
        ),
    )

    color_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Per-category intensity for Color. Multiplied by global Intensity.",
        json_schema_extra=ui_field_config(
            order=23,
            label="Color Intensity",
        ),
    )

    enable_blend: bool = Field(
        default=False,
        description="Layer compositing and mixing modes",
        json_schema_extra=ui_field_config(
            order=24,
            label="Enable Blend",
        ),
    )
    blend_shader: BlendShader = Field(
        default="screen",
        description="screen = lighten and brighten, multiply = darken and deepen, overlay = boost contrast",
        json_schema_extra=ui_field_config(
            order=26,
            label="Blend Shader",
        ),
    )

    blend_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Per-category intensity for Blend. Multiplied by global Intensity.",
        json_schema_extra=ui_field_config(
            order=27,
            label="Blend Intensity",
        ),
    )

    enable_edge: bool = Field(
        default=False,
        description="Detect and highlight edges and outlines",
        json_schema_extra=ui_field_config(
            order=28,
            label="Enable Edge",
        ),
    )
    edge_shader: EdgeShader = Field(
        default="sobel",
        description="sobel = gradient edge detection, outline = white-on-black lines, neon = glowing colored edges",
        json_schema_extra=ui_field_config(
            order=30,
            label="Edge Shader",
        ),
    )

    edge_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Per-category intensity for Edge. Multiplied by global Intensity.",
        json_schema_extra=ui_field_config(
            order=31,
            label="Edge Intensity",
        ),
    )

    enable_blur: bool = Field(
        default=False,
        description="Soften and smooth the image",
        json_schema_extra=ui_field_config(
            order=32,
            label="Enable Blur",
        ),
    )
    blur_shader: BlurShader = Field(
        default="gaussian",
        description="gaussian = soft even blur, motion = directional streak, radial = zoom blur from center",
        json_schema_extra=ui_field_config(
            order=34,
            label="Blur Shader",
        ),
    )

    blur_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Per-category intensity for Blur. Multiplied by global Intensity.",
        json_schema_extra=ui_field_config(
            order=35,
            label="Blur Intensity",
        ),
    )

    enable_generative: bool = Field(
        default=False,
        description="Procedural patterns and textures overlaid on video",
        json_schema_extra=ui_field_config(
            order=36,
            label="Enable Generative",
        ),
    )
    generative_shader: GenerativeShader = Field(
        default="noise",
        description="noise = animated static grain, pattern = scrolling sine grid, fractal = layered organic shapes",
        json_schema_extra=ui_field_config(
            order=38,
            label="Generative Shader",
        ),
    )

    generative_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Per-category intensity for Generative. Multiplied by global Intensity.",
        json_schema_extra=ui_field_config(
            order=39,
            label="Generative Intensity",
        ),
    )

    enable_atmospheric: bool = Field(
        default=False,
        description="Ambient lighting, fog, and glow effects",
        json_schema_extra=ui_field_config(
            order=40,
            label="Enable Atmospheric",
        ),
    )
    atmospheric_shader: AtmosphericShader = Field(
        default="fog",
        description="fog = haze and mist overlay, glow = pulsing highlight shimmer, bloom = bright-area light bleed",
        json_schema_extra=ui_field_config(
            order=42,
            label="Atmospheric Shader",
        ),
    )

    atmospheric_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Per-category intensity for Atmospheric. Multiplied by global Intensity.",
        json_schema_extra=ui_field_config(
            order=43,
            label="Atmospheric Intensity",
        ),
    )

    enable_utility: bool = Field(
        default=False,
        description="Basic image processing and color tools",
        json_schema_extra=ui_field_config(
            order=44,
            label="Enable Utility",
        ),
    )
    utility_shader: UtilityShader = Field(
        default="invert",
        description="invert = flip all colors, posterize = reduce to fewer tones, threshold = black and white cutoff",
        json_schema_extra=ui_field_config(
            order=46,
            label="Utility Shader",
        ),
    )

    utility_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Per-category intensity for Utility. Multiplied by global Intensity.",
        json_schema_extra=ui_field_config(
            order=47,
            label="Utility Intensity",
        ),
    )

    # ── Global effect parameters (master multipliers) ─────────────
    intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Master intensity multiplier. Scales all per-category intensities. 0 = off, 1 = normal, 2 = maximum",
        json_schema_extra=ui_field_config(
            order=50,
            label="Master Intensity",
        ),
    )

    speed: float = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        description="How fast animations move. Controls CRT roll, glitch flicker, wave scroll, etc.",
        json_schema_extra=ui_field_config(
            order=51,
            label="Speed",
        ),
    )

    scale: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Size of the effect. Affects distortion waves, blur radius, generative patterns",
        json_schema_extra=ui_field_config(
            order=52,
            label="Scale",
        ),
    )

    # ── Color parameters ───────────────────────────────────────────
    hue_shift: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Rotate the color wheel. -1 and 1 = full rotation, 0 = no change",
        json_schema_extra=ui_field_config(
            order=53,
            label="Hue Shift",
        ),
    )

    saturation: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Color richness. 0 = grayscale, 1 = normal, 2 = vivid",
        json_schema_extra=ui_field_config(
            order=54,
            label="Saturation",
        ),
    )

    brightness: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Final brightness. 0 = black, 1 = normal, 2 = blown out",
        json_schema_extra=ui_field_config(
            order=55,
            label="Brightness",
        ),
    )

    # ── Blend mode ─────────────────────────────────────────────────
    blend_mode: BlendMode = Field(
        default="normal",
        description="How the final result mixes with the original. normal = replace, screen = lighten, multiply = darken, overlay = contrast",
        json_schema_extra=ui_field_config(
            order=56,
            label="Blend Mode",
        ),
    )
