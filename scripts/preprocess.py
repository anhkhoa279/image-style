import os
from pathlib import Path

import cv2
import numpy as np

def center_crop(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    target_h, target_w = size
    if h < target_h or w < target_w:
        scale = max(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    top = max(0, (h - target_h) // 2)
    left = max(0, (w - target_w) // 2)
    return img[top : top + target_h, left : left + target_w]


def normalize_color(img: np.ndarray, mean: tuple[float, float, float] | None = None, std: tuple[float, float, float] | None = None) -> np.ndarray:
    mean = mean or (0.485 * 255, 0.456 * 255, 0.406 * 255)
    std = std or (0.229 * 255, 0.224 * 255, 0.225 * 255)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    out = (img.astype(np.float32) - mean) / std
    out = (out - out.min()) / (out.max() - out.min() + 1e-8) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def process_images(
    input_folder,
    output_folder,
    target_size=(512, 512),
    prefix="img",
    use_crop: bool = False,
    use_color_norm: bool = False,
):
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)

    # Kiểm tra
    if not in_dir.exists():
        print(f"CẢNH BÁO: Không tìm thấy thư mục {in_dir}")
        print("-> Bạn hãy tạo thư mục và bỏ ảnh vào đó trước nhé!")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    count = 1
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    print(f"\n--- Đang xử lý thư mục: {in_dir} ---")

    files = sorted([p for p in in_dir.iterdir() if p.is_file()])
    if len(files) == 0:
        print("-> Thư mục này đang TRỐNG. Hãy tải ảnh về bỏ vào đây.")
        return

    for path in files:
        if path.suffix.lower() not in valid_extensions:
            continue

        img = None
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None

        if img is None:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

        # Chuẩn hoá số kênh để ghi JPG
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Crop (center crop trước resize nếu bật)
        if use_crop:
            try:
                img = center_crop(img, target_size)
            except Exception:
                pass

        # Resize
        try:
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        except Exception:
            continue

        # Chuẩn hóa màu
        if use_color_norm:
            img_resized = normalize_color(img_resized)

        new_filename = f"{prefix}_{count:03d}.jpg"
        save_path = out_dir / new_filename
        cv2.imwrite(str(save_path), img_resized)
        count += 1

    print(f"-> HOÀN THÀNH. Tổng cộng: {count - 1} ảnh được lưu tại {out_dir}")

if __name__ == "__main__":
    # use_crop: center crop trước resize (giữ tỉ lệ khung hình quan trọng)
    # use_color_norm: chuẩn hóa màu theo mean/std ImageNet
    kw = dict(use_crop=False, use_color_norm=False)

    # 1. Xử lý ảnh Kim Hoàng
    process_images(
        input_folder="dataset/raw/kim_hoang",
        output_folder="dataset/processed/kim_hoang",
        prefix="kimhoang",
        **kw,
    )

    # 2. Xử lý ảnh Sơn Dầu
    process_images(
        input_folder="dataset/raw/son_dau",
        output_folder="dataset/processed/son_dau",
        prefix="sondau",
        **kw,
    )