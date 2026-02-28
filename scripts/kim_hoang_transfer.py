"""
Chuyển ảnh thường sang tranh Kim Hoàng (Kim Hoàng Transfer).

Đặc điểm tranh Kim Hoàng:
- Chất liệu và màu nền: Tranh in trên nền giấy màu đỏ (giấy điều), còn gọi "tranh Đỏ"
- Phong cách: Nét vẽ phóng khoáng, tự nhiên, khỏe khoắn, mang đậm nét mộc mạc
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def _apply_bold_stylization(img: np.ndarray, sigma_s: int = 50, sigma_r: float = 0.5) -> np.ndarray:
    """Stylization: nét vẽ phóng khoáng, khỏe khoắn."""
    return cv2.stylization(img, sigma_s=sigma_s, sigma_r=sigma_r)


def _apply_edge_preserving_simplify(img: np.ndarray, sigma_s: int = 60, sigma_r: float = 0.45) -> np.ndarray:
    """Đơn giản hóa ảnh giữ biên, tạo nét mộc mạc."""
    return cv2.edgePreservingFilter(img, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)


def apply_red_paper_background(
    img: np.ndarray,
    red_tint: float = 0.35,
    paper_color_bgr: tuple[int, int, int] = (45, 55, 165),
) -> np.ndarray:
    """
    Áp dụng nền giấy điều (màu đỏ) - đặc trưng tranh Kim Hoàng.
    paper_color_bgr: BGR cho giấy điều (đỏ: B thấp, G thấp, R cao).
    """
    paper = np.full_like(img, paper_color_bgr, dtype=np.uint8)
    # Blend: vùng sáng giữ nhiều màu gốc, vùng tối lộ nền đỏ hơn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blend_mask = 1 - gray  # vùng tối = blend nhiều giấy đỏ
    blend_mask = np.clip(blend_mask * red_tint, 0, 1)[:, :, np.newaxis]
    out = img.astype(np.float32) * (1 - blend_mask) + paper.astype(np.float32) * blend_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_bold_ink_edges(img: np.ndarray, strength: float = 0.25, thickness: int = 2) -> np.ndarray:
    """Nét viền đậm, khỏe khoắn như mực in trên giấy điều."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((thickness, thickness), np.uint8)
    edges = cv2.dilate(edges, kernel)
    edges_norm = edges.astype(np.float32) / 255.0
    blend = np.clip(edges_norm[:, :, np.newaxis] * strength, 0, 1)
    # Màu đen/màu tối cho nét viền
    dark = np.zeros_like(img, dtype=np.float32)
    dark[:, :] = [30, 25, 35]  # BGR: tím đen nhẹ
    out = img.astype(np.float32) * (1 - blend) + dark * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_rustic_tones(
    img: np.ndarray,
    saturation: float = 0.15,
    contrast: float = 1.08,
    warmth: float = 0.08,
) -> np.ndarray:
    """Tông mộc mạc: tăng độ bão hòa nhẹ, tương phản, ấm đất."""
    out = img.astype(np.float32)
    # Warmth nhẹ (đỏ/vàng)
    if warmth > 0:
        out[:, :, 0] = np.clip(out[:, :, 0] - warmth * 15, 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] + warmth * 12, 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] + warmth * 25, 0, 255)
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    mean = out.mean()
    out = np.clip((out - mean) * contrast + mean, 0, 255)
    return out.astype(np.uint8)


def add_paper_texture(img: np.ndarray, strength: float = 0.03) -> np.ndarray:
    """Kết cấu giấy mỏng, gợi chất liệu in trên giấy điều."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    np.random.seed(42)
    noise = np.random.randn(h, w).astype(np.float32) * strength * 40
    noise = cv2.GaussianBlur(noise, (3, 3), 0.5)
    texture = np.stack([noise] * 3, axis=-1)
    out = img.astype(np.float32) + texture
    return np.clip(out, 0, 255).astype(np.uint8)


def transfer_to_kim_hoang(
    img: np.ndarray,
    red_tint: float = 0.35,
    stylize_sigma_s: int = 50,
    stylize_sigma_r: float = 0.5,
    edge_strength: float = 0.22,
    edge_thickness: int = 2,
    saturation: float = 0.15,
    contrast: float = 1.08,
    warmth: float = 0.08,
    add_texture: bool = True,
    texture_strength: float = 0.03,
) -> np.ndarray:
    """
    Chuyển ảnh thường sang tranh Kim Hoàng.

    Đặc điểm: nền giấy đỏ (điều), nét vẽ phóng khoáng, khỏe khoắn, mộc mạc.
    """
    # 1. Stylization: nét phóng khoáng, tự nhiên
    result = _apply_bold_stylization(img, sigma_s=stylize_sigma_s, sigma_r=stylize_sigma_r)

    # 2. Đơn giản hóa nhẹ (mộc mạc)
    result = cv2.addWeighted(result, 0.7, _apply_edge_preserving_simplify(result), 0.3, 0)

    # 3. Nền giấy điều (đỏ)
    result = apply_red_paper_background(result, red_tint=red_tint)

    # 4. Nét viền đậm (mực in)
    result = apply_bold_ink_edges(result, strength=edge_strength, thickness=edge_thickness)

    # 5. Tông mộc mạc
    result = apply_rustic_tones(
        result,
        saturation=saturation,
        contrast=contrast,
        warmth=warmth,
    )

    # 6. Kết cấu giấy
    if add_texture:
        result = add_paper_texture(result, strength=texture_strength)

    return result


def process_folder(
    input_folder: str,
    output_folder: str,
    prefix: str = "kimhoang",
    **kwargs,
) -> None:
    """Xử lý toàn bộ ảnh trong thư mục."""
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)

    if not in_dir.exists():
        print(f"CANH BAO: Khong tim thay thu muc {in_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext])

    if not files:
        print("-> Thu muc rong hoac khong co anh hop le.")
        return

    count = 1
    for path in files:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        result = transfer_to_kim_hoang(img, **kwargs)
        out_path = out_dir / f"{prefix}_{count:04d}.jpg"
        cv2.imwrite(str(out_path), result)
        count += 1

    print(f"-> HOAN THANH. Da chuyen {count - 1} anh sang tranh Kim Hoang tai {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển ảnh thường sang tranh Kim Hoàng (Kim Hoàng Transfer)"
    )
    parser.add_argument("input", type=str, help="Ảnh hoặc thư mục đầu vào")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Ảnh hoặc thư mục đầu ra (mặc định: input_kimhoang)",
    )
    parser.add_argument(
        "--red-tint",
        type=float,
        default=0.35,
        help="Độ đậm nền giấy đỏ 0.2–0.5 (mặc định: 0.35)",
    )
    parser.add_argument(
        "--edge-strength",
        type=float,
        default=0.22,
        help="Cường độ nét viền mực 0.15–0.35 (mặc định: 0.22)",
    )
    parser.add_argument(
        "--edge-thickness",
        type=int,
        default=2,
        help="Độ dày nét viền 1–3 (mặc định: 2)",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=0.15,
        help="Độ bão hòa màu 0.05–0.25 (mặc định: 0.15)",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.08,
        help="Độ tương phản 1.0–1.2 (mặc định: 1.08)",
    )
    parser.add_argument(
        "--warmth",
        type=float,
        default=0.08,
        help="Tông ấm 0.05–0.15 (mặc định: 0.08)",
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Tắt kết cấu giấy",
    )
    parser.add_argument(
        "--texture-strength",
        type=float,
        default=0.03,
        help="Cường độ kết cấu giấy (mặc định: 0.03)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="kimhoang",
        help="Tiền tố tên file khi xử lý thư mục (mặc định: kimhoang)",
    )

    args = parser.parse_args()
    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.parent / f"{inp.stem}_kimhoang"

    kwargs = dict(
        red_tint=args.red_tint,
        edge_strength=args.edge_strength,
        edge_thickness=args.edge_thickness,
        saturation=args.saturation,
        contrast=args.contrast,
        warmth=args.warmth,
        add_texture=not args.no_texture,
        texture_strength=args.texture_strength,
    )

    if inp.is_file():
        img = cv2.imread(str(inp))
        if img is None:
            img = cv2.imdecode(np.fromfile(str(inp), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"LOI: Khong doc duoc anh {inp}")
            return

        result = transfer_to_kim_hoang(img, **kwargs)
        out = out.with_suffix(".jpg") if out.suffix.lower() not in {".jpg", ".jpeg", ".png"} else out
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result)
        print(f"-> Da luu tranh Kim Hoang tai {out}")

    elif inp.is_dir():
        out_dir = out if out.suffix == "" else out.parent
        process_folder(
            input_folder=str(inp),
            output_folder=str(out_dir),
            prefix=args.prefix,
            **kwargs,
        )
    else:
        print(f"LOI: Khong tim thay {inp}")


if __name__ == "__main__":
    main()
