"""
Pipeline đồng nhất: Raw → Tranh Kim Hoàng → Resize/Crop/Chuẩn hóa màu → Flip/Rotation/Color Jitter
→ Lưu tất cả vào một folder: dataset/output_kimhoang/ (hoặc tùy chỉnh)

Flow:
  raw (ảnh input)
  → Kim Hoàng transfer (nền giấy đỏ, nét phóng khoáng, mộc mạc)
  → resize, crop, chuẩn hóa màu (processed)
  → flip, rotation, color jitter (augmentation)
  → output_kimhoang/ (một folder chứa toàn bộ ảnh tranh Kim Hoàng đã xử lý + augmentation)
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
from kim_hoang_transfer import transfer_to_kim_hoang
from preprocess import center_crop, normalize_color

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TARGET_SIZE = (512, 512)


def load_image(path: Path) -> np.ndarray | None:
    """Đọc ảnh từ đường dẫn."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def process_kim_hoang_image(
    img: np.ndarray,
    target_size: tuple[int, int],
    use_crop: bool,
    use_color_norm: bool,
) -> np.ndarray:
    """Resize, crop (tùy chọn), chuẩn hóa màu (tùy chọn) trên ảnh tranh Kim Hoàng."""
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
    output_folder: str = "dataset/output_kimhoang",
    prefix: str = "kimhoang_style",
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    use_crop: bool = False,
    use_color_norm: bool = False,
    do_flip_h: bool = True,
    do_flip_v: bool = True,
    rotations: list[float] | None = None,
    do_color_jitter: bool = True,
    jitter_strength: float = 0.15,
    max_aug_per_image: int = 10,
    **kim_hoang_kw,
) -> None:
    """
    Chạy pipeline: raw → Kim Hoàng → resize/crop/color norm → augmentation → output.

    Args:
        input_folder: Thư mục ảnh raw đầu vào.
        output_folder: Thư mục đầu ra (mặc định: dataset/output_kimhoang).
        prefix: Tiền tố tên file (vd: kimhoang_style).
        target_size: Kích thước resize (mặc định 512x512).
        use_crop: Center crop trước resize.
        use_color_norm: Chuẩn hóa màu theo ImageNet.
        do_flip_h / do_flip_v: Bật flip ngang/dọc.
        rotations: Góc xoay (độ), mặc định [90, 180, 270].
        do_color_jitter: Bật color jitter.
        jitter_strength: Cường độ jitter.
        max_aug_per_image: Số lượng ảnh augmentation tối đa mỗi ảnh gốc.
        **kim_hoang_kw: Tham số truyền vào transfer_to_kim_hoang.
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    global_count = 1

    print(f"--- Pipeline Kim Hoàng: {in_dir} → {out_dir} ---")
    print(f"    Resize {target_size}, crop={use_crop}, color_norm={use_color_norm}")
    print(f"    Augmentation: flip, rotate {rotations}, color_jitter (max {max_aug_per_image}/ảnh)")

    for path in files:
        img = load_image(path)
        if img is None:
            continue

        # 1. Kim Hoàng transfer
        kim_hoang_img = transfer_to_kim_hoang(img, **kim_hoang_kw)

        # 2. Processed: resize, crop, color norm
        processed = process_kim_hoang_image(
            kim_hoang_img,
            target_size=target_size,
            use_crop=use_crop,
            use_color_norm=use_color_norm,
        )

        # 3. Augmentations từ ảnh processed
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

    print(f"-> HOÀN THÀNH. Tổng {global_count - 1} ảnh tại {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: Raw → Kim Hoàng → Resize/Crop/Color norm → Flip/Rotation/Jitter → output"
    )
    parser.add_argument(
        "input",
        type=str,
        default="dataset/raw/kim_hoang",
        nargs="?",
        help="Thư mục ảnh raw (mặc định: dataset/raw/kim_hoang)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="dataset/output_kimhoang",
        help="Thư mục đầu ra (mặc định: dataset/output_kimhoang)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="kimhoang_style",
        help="Tiền tố tên file (mặc định: kimhoang_style)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Kích thước resize (mặc định: 512)",
    )
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
        "--red-tint",
        type=float,
        default=0.35,
        help="Độ đậm nền giấy đỏ (mặc định: 0.35)",
    )
    parser.add_argument(
        "--edge-strength",
        type=float,
        default=0.22,
        help="Cường độ nét viền mực (mặc định: 0.22)",
    )

    args = parser.parse_args()
    run_pipeline(
        input_folder=args.input,
        output_folder=args.output,
        prefix=args.prefix,
        target_size=(args.size, args.size),
        use_crop=args.crop,
        use_color_norm=args.color_norm,
        do_flip_h=not args.no_flip_h,
        do_flip_v=not args.no_flip_v,
        do_color_jitter=not args.no_jitter,
        max_aug_per_image=args.max_aug,
        red_tint=args.red_tint,
        edge_strength=args.edge_strength,
    )


if __name__ == "__main__":
    main()
