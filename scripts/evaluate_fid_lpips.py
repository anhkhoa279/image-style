from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def check_deps():
    try:
        import torch
        import lpips
    except ImportError:
        print("Cài đặt: pip install torch torchvision lpips pytorch-fid")
        sys.exit(1)


def run_fid(path1: str | Path, path2: str | Path) -> float:
    path1 = Path(path1).resolve()
    path2 = Path(path2).resolve()
    if not path1.is_dir() or not path2.is_dir():
        raise FileNotFoundError(f"Cần 2 thư mục hợp lệ: {path1}, {path2}")
    out = subprocess.run(
        [sys.executable, "-m", "pytorch_fid", str(path1), str(path2)],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"pytorch_fid lỗi: {out.stderr}")
    for line in out.stdout.splitlines():
        if "FID:" in line or "FID " in line:
            parts = line.replace("FID:", " ").replace("FID", " ").split()
            for p in parts:
                try:
                    return float(p)
                except ValueError:
                    pass
    raise ValueError("Không parse được FID từ output")


def run_lpips_batch(
    path_dir1: str | Path,
    path_dir2: str | Path,
    max_pairs: int = 100,
    pair_by_index: bool = False,
) -> float:
    """LPIPS giữa 2 thư mục. pair_by_index: ghép theo thứ tự thay vì tên file."""
    import torch
    import lpips
    from PIL import Image
    from torchvision import transforms

    path_dir1 = Path(path_dir1)
    path_dir2 = Path(path_dir2)
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    list1 = sorted([p for p in path_dir1.iterdir() if p.is_file() and p.suffix.lower() in valid])
    list2 = sorted([p for p in path_dir2.iterdir() if p.is_file() and p.suffix.lower() in valid])

    if pair_by_index:
        n = min(len(list1), len(list2), max_pairs)
        pairs = list(zip(list1[:n], list2[:n]))
        if not pairs:
            raise ValueError("Không có ảnh trong 2 thư mục")
    else:
        f1 = {p.name: p for p in list1}
        f2 = {p.name: p for p in list2}
        common = list(set(f1.keys()) & set(f2.keys()))[:max_pairs]
        if not common:
            raise ValueError("Không có cặp ảnh cùng tên giữa 2 thư mục")
        pairs = [(f1[n], f2[n]) for n in common]

    tfn = transforms.ToTensor()
    to_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    loss_fn = lpips.LPIPS(net="alex")

    scores = []
    for p1, p2 in pairs:
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        t1 = to_norm(tfn(img1)).unsqueeze(0)
        t2 = to_norm(tfn(img2)).unsqueeze(0)
        with torch.no_grad():
            d = loss_fn(t1, t2).item()
        scores.append(d)

    return sum(scores) / len(scores)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Đánh giá FID và LPIPS giữa ảnh style gốc và ảnh sinh ra"
    )
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Thư mục ảnh style gốc (reference)",
    )
    parser.add_argument(
        "--gen",
        type=str,
        default=None,
        help="Thư mục ảnh sinh ra (generated)",
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["son_dau", "kim_hoang"],
        default="son_dau",
        help="Style mặc định: son_dau hoặc kim_hoang",
    )
    parser.add_argument(
        "--pair-by-index",
        action="store_true",
        help="Ghép cặp LPIPS theo thứ tự file (khi tên khác nhau)",
    )
    parser.add_argument("--max-pairs", type=int, default=50, help="Số cặp tối đa cho LPIPS")
    args = parser.parse_args()

    check_deps()

    base = Path(__file__).resolve().parent.parent
    if args.ref and args.gen:
        ref = Path(args.ref).resolve()
        gen = Path(args.gen).resolve()
    else:
        if args.style == "kim_hoang":
            ref = base / "dataset" / "processed" / "kim_hoang"
            gen = base / "dataset" / "output_kimhoang"
        else:
            ref = base / "dataset" / "processed" / "son_dau"
            gen = base / "dataset" / "output_image_style"

    if not ref.exists():
        print(f"Chạy preprocess/pipeline trước để có {ref}")
        sys.exit(1)
    if not gen.exists():
        print(f"Chạy pipeline trước để có {gen}")
        sys.exit(1)

    print(f"Style: {args.style} | Ref: {ref} | Gen: {gen}")
    print("\nFID (2 thư mục giống nhau -> gần 0):")
    try:
        fid = run_fid(ref, gen)
        print(f"  FID = {fid:.2f}")
    except Exception as e:
        print(f"  Lỗi: {e}")
        print("  Cài: pip install pytorch-fid")

    print("\nLPIPS trung bình (cặp ảnh, 0 = giống hệt):")
    try:
        lp = run_lpips_batch(
            ref, gen, max_pairs=args.max_pairs, pair_by_index=args.pair_by_index
        )
        print(f"  LPIPS = {lp:.4f}")
    except Exception as e:
        print(f"  Lỗi: {e}")
