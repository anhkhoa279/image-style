"""
Augmentation bằng PyTorch / torchvision (Deep Learning stack).
Processed → flip, rotate, color jitter → augmented.

Thay thế OpenCV bằng torchvision.transforms để toàn bộ pipeline dùng Deep Learning.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def flip_horizontal_tensor(img: torch.Tensor) -> torch.Tensor:
    """Flip ngang tensor (1,C,H,W) hoặc (C,H,W)."""
    return torch.flip(img, dims=[-1])


def flip_vertical_tensor(img: torch.Tensor) -> torch.Tensor:
    """Flip dọc tensor (1,C,H,W) hoặc (C,H,W)."""
    return torch.flip(img, dims=[-2])


def rotate_tensor(img: torch.Tensor, angle: float) -> torch.Tensor:
    """Xoay tensor (1,C,H,W) theo góc angle (độ)."""
    return transforms.functional.rotate(img, angle)


def color_jitter_tensor(
    img: torch.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> torch.Tensor:
    """Color jitter trên tensor (1,C,H,W) range [0,1]."""
    jitter = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
    )
    return jitter(img)


def get_augmented_variants(
    tensor: torch.Tensor,
    do_flip_h: bool = True,
    do_flip_v: bool = True,
    rotations: list[float] | None = None,
    do_color_jitter: bool = True,
    jitter_strength: float = 0.15,
    max_per_image: int = 10,
) -> list[torch.Tensor]:
    """
    Tạo danh sách các biến thể augmentation từ một tensor.
    Trả về list tensor, mỗi phần tử (1,C,H,W).
    """
    if rotations is None:
        rotations = [90.0, 180.0, 270.0]

    augs: list[torch.Tensor] = [tensor]

    if do_flip_h:
        augs.append(flip_horizontal_tensor(tensor))
    if do_flip_v:
        augs.append(flip_vertical_tensor(tensor))
    for angle in rotations:
        augs.append(rotate_tensor(tensor, angle))
    if do_color_jitter:
        for _ in range(2):
            augs.append(
                color_jitter_tensor(
                    tensor,
                    jitter_strength,
                    jitter_strength,
                    jitter_strength,
                )
            )

    return augs[:max_per_image]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Chuyển tensor (1,C,H,W) [0,1] sang PIL."""
    return transforms.ToPILImage()(tensor.cpu().squeeze(0).clamp(0, 1))


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
    device: torch.device | None = None,
) -> None:
    """
    Augmentation thư mục ảnh bằng PyTorch/torchvision.
    Load bằng PIL, chuyển tensor, augment, lưu.
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)

    if not in_dir.exists():
        print(f"WARNING: Input folder not found: {in_dir}")
        return

    if device is None:
        device = torch.device("cpu")

    out_dir.mkdir(parents=True, exist_ok=True)
    rotations = rotations or [90.0, 180.0, 270.0]
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext])

    if not files:
        print("-> Empty folder or no supported images.")
        return

    count = 1
    to_tensor = transforms.ToTensor()

    for path in files:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        tensor = to_tensor(img).unsqueeze(0).to(device)

        augs = get_augmented_variants(
            tensor,
            do_flip_h=do_flip_h,
            do_flip_v=do_flip_v,
            rotations=rotations,
            do_color_jitter=do_color_jitter,
            jitter_strength=jitter_strength,
            max_per_image=max_per_image,
        )

        for aug_tensor in augs:
            out_path = out_dir / f"{prefix}_{count:04d}.jpg"
            pil = tensor_to_pil(aug_tensor)
            pil.save(str(out_path))
            count += 1

    print(f"-> Done. Total {count - 1} augmented images at {out_dir}")


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    torch.manual_seed(42)

    augment_dataset(
        input_folder="dataset/processed/son_dau",
        output_folder="dataset/augmented/son_dau",
        prefix="sondau_aug",
    )
