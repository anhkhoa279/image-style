"""
Pipeline: Raw → Neural Style Transfer (sơn dầu) → Resize/Crop/Chuẩn hóa màu → Augmentation
→ Lưu vào dataset/output_image_style/

Toàn bộ bước chuyển style dùng Deep Learning (Neural Transfer, Gatys et al.).
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import cv2
import numpy as np

from augment import (
    color_jitter,
    flip_horizontal,
    flip_vertical,
    rotate,
)
from neural_style_transfer import transfer_single_to_array
from preprocess import center_crop, normalize_color

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TARGET_SIZE = (512, 512)

# Ảnh style mặc định cho sơn dầu (nếu không truyền --style)
def _default_style_path():
    base = _SCRIPT_DIR.parent
    ref = base / "dataset" / "style_ref" / "oil_style.jpg"
    if ref.exists():
        return str(ref)
    processed = base / "dataset" / "processed" / "son_dau"
    if processed.exists():
        first = sorted(processed.glob("*.jpg"))[:1]
        if first:
            return str(first[0])
    return None


def process_image(
    img: np.ndarray,
    target_size: tuple[int, int],
    use_crop: bool,
    use_color_norm: bool,
) -> np.ndarray:
    """Resize, crop (tùy chọn), chuẩn hóa màu (tùy chọn)."""
    if use_crop:
        try:
            img = center_crop(img, target_size)
        except Exception:
            pass
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    if use_color_norm:
        img = normalize_color(img)
    return img


def run_pipeline(
    input_folder: str,
    output_folder: str = "dataset/output_image_style",
    style_image: str | None = None,
    prefix: str = "sondau_style",
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    use_crop: bool = False,
    use_color_norm: bool = False,
    do_flip_h: bool = True,
    do_flip_v: bool = True,
    rotations: list[float] | None = None,
    do_color_jitter: bool = True,
    jitter_strength: float = 0.15,
    max_aug_per_image: int = 10,
    nst_imsize: int = 512,
    nst_steps: int = 300,
    style_weight: float = 1e6,
    content_weight: float = 1.0,
    device=None,
    verbose: bool = True,
) -> None:
    """
    Pipeline: raw → Neural Style Transfer (DL) → resize/crop/color norm → augmentation → output.
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    style_path = style_image or _default_style_path()
    if not style_path or not Path(style_path).exists():
        print("LỖI: Chưa có ảnh style. Đặt ảnh tại dataset/style_ref/oil_style.jpg hoặc dùng --style path/to/tranh_son_dau.jpg")
        return

    if not in_dir.exists():
        print(f"CẢNH BÁO: Không tìm thấy thư mục {in_dir}")
        return

    files = sorted(
        [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT]
    )
    if not files:
        print("-> Thư mục rỗng hoặc không có ảnh hợp lệ.")
        return

    rotations = rotations or [90.0, 180.0, 270.0]
    np.random.seed(42)

    print(f"--- Pipeline (Neural Transfer): {in_dir} → {out_dir} ---")
    print(f"    Style: {style_path}")
    print(f"    NST: size={nst_imsize}, steps={nst_steps}")
    print(f"    Resize {target_size}, crop={use_crop}, color_norm={use_color_norm}")
    print(f"    Augmentation: flip, rotate {rotations}, jitter (max {max_aug_per_image}/ảnh)")

    global_count = 1
    for idx, path in enumerate(files):
        if verbose:
            print(f"\n[{idx + 1}/{len(files)}] NST: {path.name}")

        # 1. Neural Style Transfer (Deep Learning)
        try:
            rgb = transfer_single_to_array(
                path,
                style_path,
                imsize=nst_imsize,
                num_steps=nst_steps,
                style_weight=style_weight,
                content_weight=content_weight,
                device=device,
                verbose=verbose,
            )
            oil_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"  Lỗi NST: {e}")
            continue

        # 2. Resize, crop, chuẩn hóa màu
        processed = process_image(
            oil_img,
            target_size=target_size,
            use_crop=use_crop,
            use_color_norm=use_color_norm,
        )

        # 3. Augmentation
        augs: list[np.ndarray] = [processed]
        if do_flip_h:
            augs.append(flip_horizontal(processed))
        if do_flip_v:
            augs.append(flip_vertical(processed))
        for angle in rotations:
            augs.append(rotate(processed, angle))
        if do_color_jitter:
            for _ in range(2):
                augs.append(
                    color_jitter(
                        processed,
                        jitter_strength,
                        jitter_strength,
                        jitter_strength,
                    )
                )

        for aug_img in augs[:max_aug_per_image]:
            out_path = out_dir / f"{prefix}_{global_count:05d}.jpg"
            cv2.imwrite(str(out_path), aug_img)
            global_count += 1

    print(f"\n-> HOÀN THÀNH. Tổng {global_count - 1} ảnh tại {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: Raw → Neural Style Transfer (sơn dầu) → Resize/Crop/Augment → output"
    )
    parser.add_argument(
        "input",
        type=str,
        default="dataset/raw/son_dau",
        nargs="?",
        help="Thư mục ảnh raw (mặc định: dataset/raw/son_dau)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="dataset/output_image_style",
        help="Thư mục đầu ra (mặc định: dataset/output_image_style)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Ảnh style sơn dầu tham chiếu (mặc định: dataset/style_ref/oil_style.jpg hoặc processed/son_dau)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sondau_style",
        help="Tiền tố tên file (mặc định: sondau_style)",
    )
    parser.add_argument("--size", type=int, default=512, help="Kích thước resize (mặc định: 512)")
    parser.add_argument("--crop", action="store_true", help="Bật center crop trước resize")
    parser.add_argument("--color-norm", action="store_true", help="Chuẩn hóa màu ImageNet")
    parser.add_argument("--no-flip-h", action="store_true", help="Tắt flip ngang")
    parser.add_argument("--no-flip-v", action="store_true", help="Tắt flip dọc")
    parser.add_argument("--no-jitter", action="store_true", help="Tắt color jitter")
    parser.add_argument(
        "--max-aug",
        type=int,
        default=10,
        help="Số ảnh augmentation tối đa mỗi ảnh (mặc định: 10)",
    )
    parser.add_argument(
        "--nst-size",
        type=int,
        default=512,
        help="Kích thước ảnh cho Neural Transfer (mặc định: 512, giảm nếu không có GPU)",
    )
    parser.add_argument(
        "--nst-steps",
        type=int,
        default=300,
        help="Số bước tối ưu NST (mặc định: 300)",
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1e6,
        help="Trọng số style loss NST (mặc định: 1e6)",
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=1.0,
        help="Trọng số content loss NST (mặc định: 1)",
    )
    parser.add_argument("--quiet", action="store_true", help="Ít log hơn")

    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.quiet:
        print(f"Device: {device}")

    run_pipeline(
        input_folder=args.input,
        output_folder=args.output,
        style_image=args.style,
        prefix=args.prefix,
        target_size=(args.size, args.size),
        use_crop=args.crop,
        use_color_norm=args.color_norm,
        do_flip_h=not args.no_flip_h,
        do_flip_v=not args.no_flip_v,
        do_color_jitter=not args.no_jitter,
        max_aug_per_image=args.max_aug,
        nst_imsize=args.nst_size,
        nst_steps=args.nst_steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        device=device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
