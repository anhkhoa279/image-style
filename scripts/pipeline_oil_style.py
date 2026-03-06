"""
Pipeline: Raw → Neural Style Transfer (sơn dầu) → Resize/Crop/Chuẩn hóa màu → Augmentation
→ Lưu vào dataset/output_image_style/

Toàn bộ dùng Deep Learning (PyTorch/torchvision):
- Neural Transfer: Gatys et al., VGG19 (https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- Preprocess & Augmentation: torchvision.transforms
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
import torchvision.transforms.functional as TF

from augment import get_augmented_variants, tensor_to_pil
from neural_style_transfer import transfer_single_to_tensor
from preprocess import process_image_tensor

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TARGET_SIZE = (512, 512)


def _apply_sonmai_grading(t: torch.Tensor) -> torch.Tensor:
    """
    Color grading nhẹ cho tranh sơn mài:
    - tăng contrast
    - giảm saturation xanh, nhấn mạnh đỏ/vàng
    """
    x = t.clamp(0, 1)
    # contrast / gamma mạnh hơn để tạo cảm giác bóng, nhiều lớp
    x = TF.adjust_contrast(x, 1.5)
    x = TF.adjust_gamma(x, gamma=0.85)
    # tách kênh
    r = x[:, 0:1, :, :]
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]
    # nhấn đỏ / vàng, giảm xanh / lam
    r = (r * 1.25 + 0.04).clamp(0, 1)
    g = (g * 0.85).clamp(0, 1)
    b = (b * 0.8).clamp(0, 1)
    x = torch.cat([r, g, b], dim=1)
    # saturation hơi giảm để màu trầm, sang trọng hơn
    x = TF.adjust_saturation(x, 0.85)
    return x.clamp(0, 1)


def _default_style_path():
    base = _SCRIPT_DIR.parent
    ref_dir = base / "dataset" / "style_ref"
    if ref_dir.exists():
        for name in ("oil_style.jpg", "oil_style.jpeg", "oil_style.png", "oil_style.webp"):
            ref = ref_dir / name
            if ref.exists():
                return str(ref)
    processed = base / "dataset" / "processed" / "son_dau"
    if processed.exists():
        first = sorted(processed.glob("*.jpg"))[:1]
        if first:
            return str(first[0])
    return None


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
    tv_weight: float = 0.0,
    multiscale: bool = False,
    nst_scales: list[int] | None = None,
    device=None,
    mode: str | None = None,
    verbose: bool = True,
) -> None:
    """
    Pipeline: raw → Neural Style Transfer (DL) → resize/crop/color norm → augmentation → output.
    Toàn bộ xử lý bằng PyTorch/torchvision.
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    style_path = style_image or _default_style_path()
    if not style_path or not Path(style_path).exists():
        print("ERROR: Missing style image. Put one at dataset/style_ref/oil_style.jpg or pass --style path/to/style.jpg")
        return

    if not in_dir.exists():
        print(f"WARNING: Input folder not found: {in_dir}")
        return

    files = sorted(
        [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT]
    )
    if not files:
        print("-> Empty folder or no supported images.")
        return

    rotations = rotations or [90.0, 180.0, 270.0]
    torch.manual_seed(42)

    print(f"--- Pipeline (Neural Transfer): {in_dir} -> {out_dir} ---")
    print(f"    Style: {style_path}")
    print(f"    NST: size={nst_imsize}, steps={nst_steps}")
    print(f"    Resize {target_size}, crop={use_crop}, color_norm={use_color_norm}")
    if tv_weight:
        print(f"    TV weight: {tv_weight}")
    if multiscale:
        print(f"    Multi-scale: {nst_scales if nst_scales else 'auto'}")
    print(f"    Augmentation: flip, rotate {rotations}, jitter (max {max_aug_per_image}/image)")

    global_count = 1
    for idx, path in enumerate(files):
        if verbose:
            print(f"\n[{idx + 1}/{len(files)}] NST: {path.name}")

        # 1. Neural Style Transfer (Deep Learning - Gatys et al.)
        try:
            output_tensor = transfer_single_to_tensor(
                path,
                style_path,
                imsize=nst_imsize,
                num_steps=nst_steps,
                style_weight=style_weight,
                content_weight=content_weight,
                tv_weight=tv_weight,
                multiscale=multiscale,
                scales=nst_scales,
                device=device,
                verbose=verbose,
            )
        except Exception as e:
            print(f"  NST error: {e}")
            continue

        # 2. Resize, crop, chuẩn hóa màu (PyTorch/torchvision)
        processed = process_image_tensor(
            output_tensor,
            target_size=target_size,
            use_crop=use_crop,
            use_color_norm=use_color_norm,
        )

        # Sơn mài: thêm grading màu đặc trưng (đỏ/đen/vàng, contrast cao hơn)
        if mode == "sonmai":
            processed = _apply_sonmai_grading(processed)

        # 3. Augmentation (PyTorch/torchvision)
        augs = get_augmented_variants(
            processed,
            do_flip_h=do_flip_h,
            do_flip_v=do_flip_v,
            rotations=rotations,
            do_color_jitter=do_color_jitter,
            jitter_strength=jitter_strength,
            max_per_image=max_aug_per_image,
        )

        for aug_tensor in augs:
            out_path = out_dir / f"{prefix}_{global_count:05d}.jpg"
            pil = tensor_to_pil(aug_tensor)
            pil.save(str(out_path))
            global_count += 1

    print(f"\n-> Done. Total {global_count - 1} images at {out_dir}")


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
        help="Ảnh style tham chiếu (mặc định: dataset/style_ref/oil_style.* hoặc processed/son_dau)",
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
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.0,
        help="Total Variation weight (giảm nhiễu/bệt), vd: 1e-6",
    )
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Chạy NST multi-scale (thường chất lượng sơn dầu tốt hơn)",
    )
    parser.add_argument(
        "--nst-scales",
        type=str,
        default=None,
        help="Danh sách scale NST, vd: 128,256,512 (mặc định: auto khi bật --multiscale)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["oil", "sonmai"],
        default=None,
        help="Preset tham số: 'oil' (sơn dầu), 'sonmai' (sơn mài Việt). Ghi đè một số tham số NST & output.",
    )
    parser.add_argument("--quiet", action="store_true", help="Ít log hơn")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.quiet:
        print(f"Device: {device}")

    # Parse scales từ string
    nst_scales = [int(s.strip()) for s in args.nst_scales.split(",")] if args.nst_scales else None

    # Khởi tạo giá trị từ args
    style_image = args.style
    style_weight = args.style_weight
    content_weight = args.content_weight
    tv_weight = args.tv_weight
    multiscale = args.multiscale
    prefix = args.prefix
    mode: str | None = None

    # Mặc định tách thư mục theo preset nếu output không được override
    output_root = Path(args.output)
    if args.output == "dataset/output_image_style":
        # Sơn dầu -> dataset/output_image_style/sondau
        # Sơn mài -> dataset/output_image_style/sonmai
        if args.preset == "sonmai":
            output_root = output_root / "sonmai"
        else:
            output_root = output_root / "sondau"

    # Preset cho tranh sơn mài (phong cách Việt)
    if args.preset == "sonmai":
        base = _SCRIPT_DIR.parent
        ref_dir = base / "dataset" / "style_ref"
        if style_image is None and ref_dir.exists():
            for name in (
                "sonmai_1.png",
                "sonmai_1.jpg",
                "sonmai_1.jpeg",
                "sonmai_1.webp",
            ):
                p = ref_dir / name
                if p.exists():
                    style_image = str(p)
                    break
        # Tham số gợi ý cho sơn mài: giữ khối rõ, texture mạnh, bề mặt mịn
        style_weight = 1.2e6
        content_weight = 8.0
        tv_weight = 2e-6
        multiscale = True
        if nst_scales is None:
            nst_scales = [256, 384, 512]
        if prefix == "sondau_style":
            prefix = "sonmai_style"
        mode = "sonmai"
    elif args.preset == "oil":
        mode = "oil"

    run_pipeline(
        input_folder=args.input,
        output_folder=str(output_root),
        style_image=style_image,
        prefix=prefix,
        target_size=(args.size, args.size),
        use_crop=args.crop,
        use_color_norm=args.color_norm,
        do_flip_h=not args.no_flip_h,
        do_flip_v=not args.no_flip_v,
        do_color_jitter=not args.no_jitter,
        max_aug_per_image=args.max_aug,
        nst_imsize=args.nst_size,
        nst_steps=args.nst_steps,
        style_weight=style_weight,
        content_weight=content_weight,
        tv_weight=tv_weight,
        multiscale=multiscale,
        nst_scales=nst_scales,
        device=device,
        verbose=not args.quiet,
        mode=mode,
    )


if __name__ == "__main__":
    main()
