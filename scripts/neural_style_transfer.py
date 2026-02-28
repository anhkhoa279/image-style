"""
Neural Style Transfer sử dụng PyTorch (Gatys et al.).

Chuyển phong cách nghệ thuật từ ảnh style sang ảnh content bằng Deep Learning.
Tham khảo: https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights

# VGG normalization (ImageNet)
CNN_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
CNN_NORMALIZATION_STD = [0.229, 0.224, 0.225]

CONTENT_LAYERS_DEFAULT = ["conv_4"]
STYLE_LAYERS_DEFAULT = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


class ContentLoss(nn.Module):
    """Content loss: MSE giữa feature maps của input và content image."""

    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    """Gram matrix cho style loss."""
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    """Style loss: MSE giữa Gram matrix của input và style image."""

    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    """Chuẩn hóa ảnh theo ImageNet cho VGG."""

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean.to(img.device)) / self.std.to(img.device)


def get_style_model_and_losses(
    cnn,
    normalization_mean,
    normalization_std,
    style_img,
    content_img,
    content_layers=None,
    style_layers=None,
):
    """Tạo model VGG với các layer ContentLoss và StyleLoss."""
    content_layers = content_layers or CONTENT_LAYERS_DEFAULT
    style_layers = style_layers or STYLE_LAYERS_DEFAULT

    normalization = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Cắt bỏ các layer sau content/style loss cuối
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[: (i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(
    cnn,
    normalization_mean,
    normalization_std,
    content_img,
    style_img,
    input_img,
    num_steps=300,
    style_weight=1_000_000,
    content_weight=1,
    verbose=True,
):
    """Chạy Neural Style Transfer bằng L-BFGS."""
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])
    run = [0]

    def closure():
        with torch.no_grad():
            input_img.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses) * style_weight
        content_score = sum(cl.loss for cl in content_losses) * content_weight
        loss = style_score + content_score
        loss.backward()

        run[0] += 1
        if verbose and run[0] % 50 == 0:
            print(f"  Step {run[0]}: Style Loss={style_score.item():.4f} Content Loss={content_score.item():.4f}")

        return style_score + content_score

    while run[0] <= num_steps:
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def image_loader(image_path, size, device):
    """Load ảnh và chuyển thành tensor."""
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def tensor_to_pil(tensor):
    """Chuyển tensor về PIL Image."""
    image = tensor.cpu().clone().squeeze(0)
    image = image.clamp(0, 1)
    unloader = transforms.ToPILImage()
    return unloader(image)


def transfer_single(
    content_path,
    style_path,
    output_path,
    imsize=512,
    num_steps=300,
    style_weight=1_000_000,
    content_weight=1,
    device=None,
):
    """Chuyển style cho một ảnh."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
    mean = torch.tensor(CNN_NORMALIZATION_MEAN).to(device)
    std = torch.tensor(CNN_NORMALIZATION_STD).to(device)

    content_img = image_loader(content_path, imsize, device)
    style_img = image_loader(style_path, imsize, device)

    if content_img.size() != style_img.size():
        style_img = transforms.functional.resize(
            style_img, content_img.shape[-2:], antialias=True
        )

    input_img = content_img.clone()

    output = run_style_transfer(
        cnn, mean, std,
        content_img, style_img, input_img,
        num_steps=num_steps,
        style_weight=style_weight,
        content_weight=content_weight,
    )

    out_img = tensor_to_pil(output)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        output_path = output_path.with_suffix(".jpg")
    out_img.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Neural Style Transfer (Gatys et al.) - Deep Learning"
    )
    parser.add_argument(
        "content",
        type=str,
        help="Ảnh content (hoặc thư mục chứa ảnh)",
    )
    parser.add_argument(
        "style",
        type=str,
        help="Ảnh style tham chiếu",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="File/thư mục output (mặc định: content_styled.jpg hoặc output_dir/)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Kích thước ảnh (mặc định: 512, nhỏ hơn nếu không có GPU)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Số bước tối ưu (mặc định: 300)",
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1e6,
        help="Trọng số style loss (mặc định: 1e6)",
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=1.0,
        help="Trọng số content loss (mặc định: 1)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="nst",
        help="Tiền tố tên file khi xử lý thư mục (mặc định: nst)",
    )

    args = parser.parse_args()
    content_path = Path(args.content)
    style_path = Path(args.style)

    if not style_path.exists():
        print(f"LỖI: Không tìm thấy ảnh style: {style_path}")
        sys.exit(1)

    try:
        import torch
    except ImportError:
        print("Cần cài PyTorch: pip install torch torchvision")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if content_path.is_file():
        out = args.output or content_path.parent / f"{content_path.stem}_styled.jpg"
        print(f"Content: {content_path}")
        print(f"Style: {style_path}")
        print(f"Output: {out}")
        transfer_single(
            content_path, style_path, out,
            imsize=args.size,
            num_steps=args.steps,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            device=device,
        )
        print(f"-> Đã lưu tại {out}")

    elif content_path.is_dir():
        out_dir = Path(args.output) if args.output else content_path.parent / "output_neural_style"
        out_dir.mkdir(parents=True, exist_ok=True)

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted([f for f in content_path.iterdir() if f.suffix.lower() in valid_ext])

        if not files:
            print(f"LỖI: Không có ảnh trong {content_path}")
            sys.exit(1)

        print(f"Content folder: {content_path} ({len(files)} ảnh)")
        print(f"Style: {style_path}")
        print(f"Output: {out_dir}")

        for i, f in enumerate(files):
            out_name = f"{args.prefix}_{i+1:05d}.jpg"
            out_path = out_dir / out_name
            print(f"\n[{i+1}/{len(files)}] {f.name} -> {out_name}")
            transfer_single(
                f, style_path, out_path,
                imsize=args.size,
                num_steps=args.steps,
                style_weight=args.style_weight,
                content_weight=args.content_weight,
                device=device,
            )
        print(f"\n-> Đã lưu {len(files)} ảnh tại {out_dir}")

    else:
        print(f"LỖI: Không tìm thấy {content_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
