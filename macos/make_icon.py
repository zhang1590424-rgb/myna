from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
APP_RESOURCES = ROOT / "Myna.app" / "Contents" / "Resources"
WEB_ASSETS = ROOT / "web" / "assets"


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size, size), radius=radius, fill=255)
    return mask


def gradient(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size))
    px = image.load()
    for y in range(size):
        for x in range(size):
            nx = x / (size - 1)
            ny = y / (size - 1)
            # 极深冷灰渐变，右上角略亮，无彩色高光
            t = nx * 0.4 + (1 - ny) * 0.6
            base = (
                int(14 + 18 * t),
                int(17 + 22 * t),
                int(23 + 28 * t),
                255,
            )
            px[x, y] = base
    return image


def draw_icon(size: int) -> Image.Image:
    scale = size / 1024
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        tuple(int(v * scale) for v in (78, 92, 958, 984)),
        radius=int(238 * scale),
        fill=(0, 0, 0, 80),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(int(24 * scale)))
    canvas.alpha_composite(shadow)

    mask = rounded_mask(size, int(224 * scale))
    body = gradient(size)
    body.putalpha(mask)
    canvas.alpha_composite(body)

    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        tuple(int(v * scale) for v in (22, 22, 1002, 1002)),
        radius=int(214 * scale),
        outline=(255, 255, 255, 52),
        width=max(1, int(8 * scale)),
    )
    draw.arc(
        tuple(int(v * scale) for v in (96, 70, 928, 720)),
        194,
        340,
        fill=(255, 255, 255, 50),
        width=max(1, int(13 * scale)),
    )

    glyph_shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glyph_shadow_draw = ImageDraw.Draw(glyph_shadow)
    m_points = [(296, 684), (296, 380), (512, 590), (728, 380), (728, 684)]
    m_scaled = [(int(x * scale), int(y * scale)) for x, y in m_points]
    glyph_shadow_draw.line(
        m_scaled,
        fill=(0, 0, 0, 96),
        width=int(102 * scale),
        joint="curve",
    )
    glyph_shadow = glyph_shadow.filter(ImageFilter.GaussianBlur(int(10 * scale)))
    overlay.alpha_composite(glyph_shadow)

    draw = ImageDraw.Draw(overlay)
    draw.line(
        m_scaled,
        fill=(240, 242, 248, 255),
        width=int(84 * scale),
        joint="curve",
    )

    canvas.alpha_composite(overlay)
    return canvas


def save_web_assets(master: Image.Image) -> None:
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    for size in (32, 192, 512):
        master.resize((size, size), Image.Resampling.LANCZOS).save(
            WEB_ASSETS / f"myna-icon-{size}.png"
        )
    master.save(
        WEB_ASSETS / "favicon.ico",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64)],
    )


def save_icns(master: Image.Image) -> None:
    APP_RESOURCES.mkdir(parents=True, exist_ok=True)
    sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "MynaIcon.iconset"
        iconset.mkdir()
        for name, size in sizes.items():
            master.resize((size, size), Image.Resampling.LANCZOS).save(iconset / name)
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(APP_RESOURCES / "MynaIcon.icns")],
            check=True,
        )


def main() -> int:
    if shutil.which("iconutil") is None:
        print("iconutil is required on macOS", file=sys.stderr)
        return 1
    master = draw_icon(1024)
    save_web_assets(master)
    save_icns(master)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
