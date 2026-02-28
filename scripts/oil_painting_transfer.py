"""
Chuyển ảnh thường sang tranh sơn dầu (Oil Painting Transfer).

Đặc điểm tranh sơn dầu:
- Chất liệu sơn dầu, mảng khối 3D (impasto, nét cọ rõ)
- Màu sắc: Ánh sáng tinh tế, tông ấm áp hoặc đượm buồn hoài niệm
- Chất liệu: Độ bóng, hiệu ứng chiều sâu trên vải canvas
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def _apply_oil_painting(img: np.ndarray, size: int = 5, dyn_ratio: int = 1) -> np.ndarray:
    """oilPainting (size/dyn nhỏ = ít blocky; size/dyn lớn = mosaic)."""
    try:
        return cv2.xphoto.oilPainting(img, size, dyn_ratio, cv2.COLOR_BGR2Lab)
    except AttributeError:
        return img


def _apply_painterly_stylization(img: np.ndarray, sigma_s: int = 70, sigma_r: float = 0.4) -> np.ndarray:
    """Stylization: chuyển tiếp màu mượt, soft edges như tranh sơn dầu."""
    return cv2.stylization(img, sigma_s=sigma_s, sigma_r=sigma_r)


def apply_oil_painting_hybrid(
    img: np.ndarray,
    brush_size: int = 7,
    dyn_ratio: int = 1,
    oil_blend: float = 0.60,
) -> np.ndarray:
    """
    Hybrid: stylization (mượt) + oil painting (nét cọ rõ).
    Reference: nét cọ prominent, soft blending, không blocky.
    """
    stylized = _apply_painterly_stylization(img)
    try:
        oiled = _apply_oil_painting(img, size=brush_size, dyn_ratio=dyn_ratio)
        out = cv2.addWeighted(stylized, 1 - oil_blend, oiled, oil_blend, 0)
    except Exception:
        out = stylized
    return out


def _apply_impasto_relief(img, strength=0.18):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gradient field (hướng cấu trúc ảnh)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    # Emboss theo hướng cấu trúc
    emboss = img.astype(np.float32)
    for c in range(3):
        emboss[:,:,c] += mag * strength * 255

    return np.clip(emboss, 0, 255).astype(np.uint8)


def apply_warm_melancholic_tones(
    img: np.ndarray,
    warmth: float = 0.14,
    saturation_shift: float = 0.05,
    nostalgia: float = 0.09,
    shadow_richness: float = 0.08,
    green_warmth: float = 0.06,
) -> np.ndarray:
    """
    Tông ấm áp / hoài niệm theo reference: ánh sáng tinh tế, bóng đổ sâu (nâu/tím),
    xanh lá có tông vàng, màu phong phú.
    """
    out = img.astype(np.float32)
    # Tông ấm tổng thể
    if warmth != 0:
        out[:, :, 0] = np.clip(out[:, :, 0] - warmth * 22, 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] + warmth * 18, 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] + warmth * 38, 0, 255)

    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    # Xanh lá thêm tông vàng (H 35-85: giảm H = shift sang vàng)
    if green_warmth > 0:
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        green_mask = ((h >= 35) & (h <= 85)) & (s > 30)
        h_warm = np.clip(h - green_warmth * 20, 0, 180)  # shift sang vàng
        hsv[:, :, 0] = np.where(green_mask, h_warm, h)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation_shift), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # Nostalgia: vùng tối thêm sepia (nâu ấm)
    gray = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    dark_mask = np.clip((1 - gray) ** 1.2, 0, 1)  # vùng tối nhấn mạnh
    if nostalgia > 0:
        sepia_b, sepia_g, sepia_r = 65, 88, 118
        blend = np.clip(nostalgia * dark_mask, 0, 1)
        out[:, :, 0] = out[:, :, 0] * (1 - blend) + sepia_b * blend
        out[:, :, 1] = out[:, :, 1] * (1 - blend) + sepia_g * blend
        out[:, :, 2] = out[:, :, 2] * (1 - blend) + sepia_r * blend

    # Shadow richness: bóng đổ thêm nâu tím (reference: darker browns, purplish hints)
    if shadow_richness > 0:
        blend = np.clip(shadow_richness * dark_mask, 0, 1)[:, :, np.newaxis]
        tint = np.array([95, 78, 108], dtype=np.float32)  # tím-nâu nhẹ
        out = out * (1 - blend) + (out * 0.7 + tint * 0.3) * blend

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_brightness(img: np.ndarray, gain: float = 1.08, lift: float = 8) -> np.ndarray:
    """Tăng độ sáng giống ảnh mẫu (daylight, bright)."""
    if gain <= 1 and lift <= 0:
        return img
    out = img.astype(np.float32) * gain + lift
    return np.clip(out, 0, 255).astype(np.uint8)


def add_atmospheric_depth(img: np.ndarray, vignette: float = 0.25) -> np.ndarray:
    """
    Thêm chiều sâu không khí: vignette nhẹ (tối viền) gợi canvas/hoài niệm.
    """
    if vignette <= 0:
        return img
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    mask = 1 - vignette * (dist / max_dist) ** 2
    mask = np.clip(mask, 0, 1)[:, :, np.newaxis]
    out = img.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def add_canvas_texture(img: np.ndarray, strength: float = 0.04) -> np.ndarray:
    """Kết cấu vải canvas nhẹ, gợi chất liệu sơn dầu trên vải."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    x = np.linspace(0, np.pi * 12, w)
    y = np.linspace(0, np.pi * 12, h)
    xv, yv = np.meshgrid(x, y)
    weave = (np.sin(xv) + np.sin(yv)) * 0.5
    weave = cv2.normalize(weave, None, -1, 1, cv2.NORM_MINMAX).astype(np.float32)
    texture = np.stack([weave] * 3, axis=-1) * strength * 80
    out = img.astype(np.float32) + texture
    return np.clip(out, 0, 255).astype(np.uint8)


def add_brushstroke_texture(img: np.ndarray, strength: float = 0.04) -> np.ndarray:
    """Kết cấu nét cọ theo cấu trúc ảnh, tăng cảm giác impasto."""
    if strength <= 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    stroke = np.stack([mag * 255] * 3, axis=-1)
    out = img.astype(np.float32) + stroke * strength
    return np.clip(out, 0, 255).astype(np.uint8)

def transfer_to_oil_painting(
    img: np.ndarray,
    brush_size: int = 7,
    dyn_ratio: int = 1,
    oil_blend: float = 0.60,
    warmth: float = 0.14,
    nostalgia: float = 0.06,
    shadow_richness: float = 0.05,
    green_warmth: float = 0.07,
    impasto_strength: float = 0.15,
    vignette: float = 0.15,
    brightness: float = 1.08,
    add_texture: bool = True,
    texture_strength: float = 0.045,
    brushstroke_strength: float = 0.04,
) -> np.ndarray:
    """
    Chuyển ảnh thường sang tranh sơn dầu (reference: Chùa 1 Cột, Hồ Hoàn Kiếm,
    Kinh thành Huế, Làng cổ Đường Lâm).

    Đặc điểm: nét cọ rõ, soft blending, tông ấm đất, impasto, chiều sâu.
    """
    # 1. Hybrid: stylization + oil painting → nét cọ rõ, mượt
    result = apply_oil_painting_hybrid(
        img, brush_size=brush_size, dyn_ratio=dyn_ratio, oil_blend=oil_blend
    )

    # 2. Khối sơn nổi (impasto)
    result = _apply_impasto_relief(result, strength=impasto_strength)

    # 3. Soft color diffusion (gợi blend màu sơn dầu)
    blur = cv2.GaussianBlur(result, (0, 0), 2)
    result = cv2.addWeighted(result, 0.92, blur, 0.08, 0)

    # 4. Tông màu ấm, hoài niệm, bóng đổ sâu, xanh lá vàng
    result = apply_warm_melancholic_tones(
        result,
        warmth=warmth,
        nostalgia=nostalgia,
        shadow_richness=shadow_richness,
        green_warmth=green_warmth,
    )

    # 5. Chiều sâu (vignette nhẹ)
    result = add_atmospheric_depth(result, vignette=vignette)

    # 6. Tăng độ sáng (giống ảnh mẫu: bright daylight)
    result = apply_brightness(result, gain=brightness, lift=6)

    # 7. Kết cấu canvas + nét cọ
    if add_texture:
        result = add_canvas_texture(result, strength=texture_strength)
        result = add_brushstroke_texture(result, strength=brushstroke_strength)

    return result


def process_folder(
    input_folder: str,
    output_folder: str,
    prefix: str = "oil",
    brush_size: int = 7,
    dyn_ratio: int = 1,
    oil_blend: float = 0.60,
    warmth: float = 0.14,
    nostalgia: float = 0.06,
    shadow_richness: float = 0.05,
    green_warmth: float = 0.07,
    impasto_strength: float = 0.15,
    vignette: float = 0.15,
    brightness: float = 1.08,
    add_texture: bool = True,
    texture_strength: float = 0.045,
    brushstroke_strength: float = 0.04,
) -> None:
    """Xử lý toàn bộ ảnh trong thư mục."""
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)

    if not in_dir.exists():
        print(f"CẢNH BÁO: Không tìm thấy thư mục {in_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext])

    if not files:
        print("-> Thư mục rỗng hoặc không có ảnh hợp lệ.")
        return

    count = 1
    for path in files:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        result = transfer_to_oil_painting(
            img,
            brush_size=brush_size,
            dyn_ratio=dyn_ratio,
            oil_blend=oil_blend,
            warmth=warmth,
            nostalgia=nostalgia,
            shadow_richness=shadow_richness,
            green_warmth=green_warmth,
            impasto_strength=impasto_strength,
            vignette=vignette,
            brightness=brightness,
            add_texture=add_texture,
            texture_strength=texture_strength,
            brushstroke_strength=brushstroke_strength,
        )

        out_path = out_dir / f"{prefix}_{count:04d}.jpg"
        cv2.imwrite(str(out_path), result)
        count += 1

    print(f"-> HOÀN THÀNH. Đã chuyển {count - 1} ảnh sang tranh sơn dầu tại {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển ảnh thường sang tranh sơn dầu (Oil Painting Transfer)"
    )
    parser.add_argument("input", type=str, help="Ảnh hoặc thư mục đầu vào")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Ảnh hoặc thư mục đầu ra (mặc định: input_oil)",
    )
    parser.add_argument(
        "--brush-size",
        type=int,
        default=7,
        help="Kích thước nét cọ 6–8 (mặc định: 7)",
    )
    parser.add_argument(
        "--dyn-ratio",
        type=int,
        default=1,
        help="Dynamic range 1–2, giữ 1 tránh blocky (mặc định: 1)",
    )
    parser.add_argument(
        "--oil-blend",
        type=float,
        default=0.60,
        help="Tỷ lệ oil trong hybrid 0.5–0.65 (mặc định: 0.60)",
    )
    parser.add_argument(
        "--warmth",
        type=float,
        default=0.14,
        help="Độ ấm tông màu 0.10–0.18 (mặc định: 0.14)",
    )
    parser.add_argument(
        "--nostalgia",
        type=float,
        default=0.06,
        help="Tông sepia vùng tối 0.04–0.10 (mặc định: 0.06)",
    )
    parser.add_argument(
        "--shadow-richness",
        type=float,
        default=0.05,
        help="Bóng đổ sâu 0.03–0.08 (mặc định: 0.05)",
    )
    parser.add_argument(
        "--green-warmth",
        type=float,
        default=0.07,
        help="Xanh lá thêm tông vàng 0.03–0.10 (mặc định: 0.07)",
    )
    parser.add_argument(
        "--impasto",
        type=float,
        default=0.15,
        help="Độ nổi khối sơn 3D (impasto) 0.10–0.20 (mặc định: 0.15)",
    )
    parser.add_argument(
        "--vignette",
        type=float,
        default=0.15,
        help="Độ tối viền 0.10–0.22, thấp = sáng hơn (mặc định: 0.15)",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=1.08,
        help="Độ sáng 1.0–1.2 (mặc định: 1.08)",
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Tắt kết cấu canvas",
    )
    parser.add_argument(
        "--texture-strength",
        type=float,
        default=0.045,
        help="Cường độ kết cấu canvas (mặc định: 0.045)",
    )
    parser.add_argument(
        "--brushstroke-strength",
        type=float,
        default=0.04,
        help="Cường độ kết cấu nét cọ (mặc định: 0.04)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="oil",
        help="Tiền tố tên file khi xử lý thư mục (mặc định: oil)",
    )

    args = parser.parse_args()
    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.parent / f"{inp.stem}_oil"

    if inp.is_file():
        img = cv2.imread(str(inp))
        if img is None:
            img = cv2.imdecode(np.fromfile(str(inp), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"LỖI: Không đọc được ảnh {inp}")
            return

        result = transfer_to_oil_painting(
            img,
            brush_size=args.brush_size,
            dyn_ratio=args.dyn_ratio,
            oil_blend=args.oil_blend,
            warmth=args.warmth,
            nostalgia=args.nostalgia,
            shadow_richness=args.shadow_richness,
            green_warmth=args.green_warmth,
            impasto_strength=args.impasto,
            vignette=args.vignette,
            brightness=args.brightness,
            add_texture=not args.no_texture,
            texture_strength=args.texture_strength,
            brushstroke_strength=args.brushstroke_strength,
        )

        out = out.with_suffix(".jpg") if out.suffix.lower() not in {".jpg", ".jpeg", ".png"} else out
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result)
        print(f"-> Đã lưu tranh sơn dầu tại {out}")

    elif inp.is_dir():
        out_dir = out if out.suffix == "" else out.parent
        process_folder(
            input_folder=str(inp),
            output_folder=str(out_dir),
            prefix=args.prefix,
            brush_size=args.brush_size,
            dyn_ratio=args.dyn_ratio,
            oil_blend=args.oil_blend,
            warmth=args.warmth,
            nostalgia=args.nostalgia,
            shadow_richness=args.shadow_richness,
            green_warmth=args.green_warmth,
            impasto_strength=args.impasto,
            vignette=args.vignette,
            brightness=args.brightness,
            add_texture=not args.no_texture,
            texture_strength=args.texture_strength,
            brushstroke_strength=args.brushstroke_strength,
        )
    else:
        print(f"LỖI: Không tìm thấy {inp}")


if __name__ == "__main__":
    main()
