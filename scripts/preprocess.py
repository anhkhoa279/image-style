"""
Preprocess ảnh bằng PyTorch / torchvision (Deep Learning stack).
Raw → Resize, CenterCrop, Normalize → processed.

Thay thế OpenCV bằng torchvision.transforms để toàn bộ pipeline dùng Deep Learning.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# ImageNet normalization (dùng cho VGG và chuẩn hóa màu)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def center_crop(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Center crop tensor (C,H,W) hoặc (1,C,H,W)."""
    if img.dim() == 4:
        img = img.squeeze(0)
    c, h, w = img.shape
    target_h, target_w = size
    if h < target_h or w < target_w:
        scale = max(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = transforms.functional.resize(
            img.unsqueeze(0), (new_h, new_w), antialias=True
        ).squeeze(0)
        h, w = new_h, new_w
    top = max(0, (h - target_h) // 2)
    left = max(0, (w - target_w) // 2)
    return img[:, top : top + target_h, left : left + target_w]


def normalize_color_tensor(img: torch.Tensor) -> torch.Tensor:
    """
    Chuẩn hóa màu theo ImageNet, rồi scale về [0,1] để lưu ảnh.
    Input: tensor (C,H,W) hoặc (1,C,H,W), range [0,1].
    """
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(-1, 1, 1)
    out = (img - mean) / std
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return out.clamp(0, 1)


def get_preprocess_transforms(
    target_size: tuple[int, int] = (512, 512),
    use_crop: bool = False,
) -> transforms.Compose:
    """
    Tạo pipeline transforms PyTorch cho preprocess.
    use_crop: Resize (shortest edge) rồi CenterCrop để giữ tỷ lệ.
    """
    tfs = []
    if use_crop:
        # Resize shortest edge = target, rồi center crop
        tfs.append(transforms.Resize(max(target_size), antialias=True))
        tfs.append(transforms.CenterCrop(target_size))
    else:
        tfs.append(transforms.Resize(target_size, antialias=True))
    tfs.append(transforms.ToTensor())
    return transforms.Compose(tfs)


def process_image_tensor(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
    use_crop: bool,
    use_color_norm: bool,
) -> torch.Tensor:
    """
    Resize, crop, chuẩn hóa màu cho tensor (1,C,H,W) từ NST.
    Dùng trong pipeline - toàn bộ PyTorch.
    """
    if use_crop:
        # Scale up nếu cần rồi center crop
        c, h, w = tensor.shape[1], tensor.shape[2], tensor.shape[3]
        th, tw = target_size
        if h < th or w < tw:
            scale = max(th / h, tw / w)
            new_h, new_w = int(h * scale), int(w * scale)
            tensor = transforms.functional.resize(
                tensor, (new_h, new_w), antialias=True
            )
        tensor = transforms.functional.center_crop(tensor, target_size)
    tensor = transforms.functional.resize(tensor, target_size, antialias=True)
    if use_color_norm:
        tensor = normalize_color_tensor(tensor)
    return tensor


def tensor_to_pil_save(tensor: torch.Tensor, path: Path) -> None:
    """Lưu tensor (1,C,H,W) [0,1] thành file ảnh."""
    img = tensor.cpu().clone().squeeze(0).clamp(0, 1)
    pil = transforms.ToPILImage()(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(str(path))


def process_images(
    input_folder,
    output_folder,
    target_size=(512, 512),
    prefix="img",
    use_crop: bool = False,
    use_color_norm: bool = False,
    device=None,
):
    """
    Preprocess thư mục ảnh bằng PyTorch/torchvision.
    Raw → Resize/Crop/Normalize → processed.
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)

    if not in_dir.exists():
        print(f"WARNING: Input folder not found: {in_dir}")
        print("-> Create the folder and put images inside first.")
        return

    if device is None:
        device = torch.device("cpu")

    out_dir.mkdir(parents=True, exist_ok=True)

    count = 1
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    print(f"\n--- Processing folder (PyTorch): {in_dir} ---")

    files = sorted([p for p in in_dir.iterdir() if p.is_file()])
    if len(files) == 0:
        print("-> Folder is empty.")
        return

    t = get_preprocess_transforms(target_size, use_crop)

    for path in files:
        if path.suffix.lower() not in valid_extensions:
            continue

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        tensor = t(img).unsqueeze(0).to(device)

        if use_color_norm:
            tensor = normalize_color_tensor(tensor)

        new_filename = f"{prefix}_{count:03d}.jpg"
        save_path = out_dir / new_filename
        tensor_to_pil_save(tensor, save_path)
        count += 1

    print(f"-> Done. Total: {count - 1} images saved to {out_dir}")


if __name__ == "__main__":
    # Preprocess cho tranh sơn dầu
    process_images(
        input_folder="dataset/raw/son_dau",
        output_folder="dataset/processed/son_dau",
        prefix="sondau",
        use_crop=False,
        use_color_norm=False,
    )

    # Preprocess cho tranh sơn mài (nếu có thư mục raw/son_mai)
    process_images(
        input_folder="dataset/raw/son_mai",
        output_folder="dataset/processed/son_mai",
        prefix="sonmai",
        use_crop=False,
        use_color_norm=False,
    )
