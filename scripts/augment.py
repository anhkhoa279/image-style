from pathlib import Path

import cv2
import numpy as np


def flip_horizontal(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def flip_vertical(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 0)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def color_jitter(
    img: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> np.ndarray:
    out = img.astype(np.float32) / 255.0

    b = 1.0 + np.random.uniform(-brightness, brightness)
    out = np.clip(out * b, 0, 1)

    c = 1.0 + np.random.uniform(-contrast, contrast)
    mean = out.mean()
    out = np.clip((out - mean) * c + mean, 0, 1)

    hsv = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    s = 1.0 + np.random.uniform(-saturation, saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR) / 255.0

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def augment_dataset(
    input_folder: str,
    output_folder: str,
    prefix: str = "aug",
    do_flip_h: bool = True,
    do_flip_v: bool = True,
    rotations: list[float] | None = None,
    do_color_jitter: bool = True,
    jitter_strength: float = 0.15,
    max_per_image: int = 10,
) -> None:

    in_dir = Path(input_folder)
    out_dir = Path(output_folder)

    if not in_dir.exists():
        print(f"CẢNH BÁO: Không tìm thấy thư mục {in_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rotations = rotations or [90, 180, 270]
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext])

    if not files:
        print("-> Thư mục rỗng hoặc không có ảnh hợp lệ.")
        return

    count = 1
    np.random.seed(42)

    for path in files:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        augs: list[np.ndarray] = [img]

        if do_flip_h:
            augs.append(flip_horizontal(img))
        if do_flip_v:
            augs.append(flip_vertical(img))
        for angle in rotations:
            augs.append(rotate(img, angle))
        if do_color_jitter:
            for _ in range(2):
                augs.append(color_jitter(img, jitter_strength, jitter_strength, jitter_strength))

        for aug_img in augs[:max_per_image]:
            out_path = out_dir / f"{prefix}_{count:04d}.jpg"
            cv2.imwrite(str(out_path), aug_img)
            count += 1

    print(f"-> HOÀN THÀNH. Tổng {count - 1} ảnh augmentation tại {out_dir}")


if __name__ == "__main__":
    augment_dataset(
        input_folder="dataset/processed/son_dau",
        output_folder="dataset/augmented/son_dau",
        prefix="sondau_aug",
    )
    augment_dataset(
        input_folder="dataset/processed/kim_hoang",
        output_folder="dataset/augmented/kim_hoang",
        prefix="kimhoang_aug",
    )
