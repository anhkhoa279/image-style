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


def run_lpips_batch(path_dir1: str | Path, path_dir2: str | Path, max_pairs: int = 100) -> float:
    import torch
    import lpips
    from PIL import Image
    from torchvision import transforms

    path_dir1 = Path(path_dir1)
    path_dir2 = Path(path_dir2)
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    f1 = {p.name: p for p in path_dir1.iterdir() if p.is_file() and p.suffix.lower() in valid}
    f2 = {p.name: p for p in path_dir2.iterdir() if p.is_file() and p.suffix.lower() in valid}
    common = list(set(f1.keys()) & set(f2.keys()))[:max_pairs]
    if not common:
        raise ValueError("Không có cặp ảnh cùng tên giữa 2 thư mục")

    tfn = transforms.ToTensor()
    to_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    loss_fn = lpips.LPIPS(net="alex")

    scores = []
    for name in common:
        img1 = Image.open(f1[name]).convert("RGB")
        img2 = Image.open(f2[name]).convert("RGB")
        t1 = to_norm(tfn(img1)).unsqueeze(0)
        t2 = to_norm(tfn(img2)).unsqueeze(0)
        with torch.no_grad():
            d = loss_fn(t1, t2).item()
        scores.append(d)

    return sum(scores) / len(scores)


if __name__ == "__main__":
    check_deps()

    base = Path(__file__).resolve().parent.parent
    ref = base / "dataset" / "processed" / "son_dau"
    gen = base / "dataset" / "processed" / "son_dau"

    if not ref.exists():
        print("Chạy preprocess trước để có dataset/processed/son_dau")
        sys.exit(1)

    print("FID (2 thư mục giống nhau -> gần 0):")
    try:
        fid = run_fid(ref, gen)
        print(f"  FID = {fid:.2f}")
    except Exception as e:
        print(f"  Lỗi: {e}")
        print("  Cài: pip install pytorch-fid")

    print("\nLPIPS trung bình (cặp ảnh cùng tên, 0 = giống hệt):")
    try:
        lp = run_lpips_batch(ref, gen, max_pairs=50)
        print(f"  LPIPS = {lp:.4f}")
    except Exception as e:
        print(f"  Lỗi: {e}")
