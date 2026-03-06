# Project_Art_Style

Repo dùng **Deep Learning (Neural Style Transfer)** để chuyển ảnh thường sang **tranh sơn dầu**, kèm tiền xử lý và augmentation, phục vụ dataset nghiên cứu nhận diện/phân loại tranh.

Toàn bộ bước chuyển style dùng **Neural Transfer** (Gatys et al., VGG19), không dùng OpenCV stylization.

## Cấu trúc thư mục

- **`dataset/raw/son_dau/`**: ảnh gốc đầu vào
- **`dataset/processed/son_dau/`**: ảnh đã resize, chuẩn hóa (preprocess)
- **`dataset/augmented/son_dau/`**: ảnh augmentation từ processed
- **`dataset/style_ref/`**: ảnh style tham chiếu (ví dụ `oil_style.jpg`) cho Neural Transfer
- **`dataset/output_image_style/`**: ảnh đầu ra pipeline (raw → NST → resize/crop/color norm → augmentation)

## Scripts

| Script | Mô tả |
|--------|--------|
| `scripts/preprocess.py` | Resize, chuẩn hóa màu: raw → processed/son_dau |
| `scripts/augment.py` | Augmentation: processed → augmented/son_dau |
| `scripts/neural_style_transfer.py` | Neural Style Transfer (Gatys, VGG19): content + style → output |
| `scripts/pipeline_oil_style.py` | **Pipeline chính**: raw → **NST** → resize/crop/color norm → augmentation → output_image_style |
| `scripts/evaluate_fid_lpips.py` | Đánh giá FID và LPIPS (ref vs generated) |

## Cài đặt

```bash
python -m pip install -r requirements.txt
```

Cần: `torch`, `torchvision`, `numpy`, `Pillow`. Đánh giá thêm: `pytorch-fid`, `lpips`. Toàn bộ dùng PyTorch/torchvision (không dùng OpenCV).

## Luồng chính: Pipeline sơn dầu (Neural Transfer)

Pipeline dùng **Neural Style Transfer** (Deep Learning) để chuyển mỗi ảnh raw sang phong cách sơn dầu, sau đó resize/crop/chuẩn hóa màu và augmentation.

### 1. Ảnh style tham chiếu

Đặt **một ảnh tranh sơn dầu** làm style reference:

- **Mặc định**: `dataset/style_ref/oil_style.jpg` (hoặc `.png/.jpeg/.webp`)  
- Hoặc chỉ định bằng `--style path/to/tranh_son_dau.jpg`.

Gợi ý để ra đúng “sơn dầu Việt / mảng khối 3D”: dùng **ảnh style là tranh sơn dầu thật** (có vệt cọ, khối sáng-tối rõ). Nếu dùng ảnh chụp bình thường làm style thì kết quả sẽ giống “lọc mờ” hơn là chất liệu sơn dầu.

### 2. Chạy pipeline

```bash
# Mặc định (sơn dầu): input=dataset/raw/son_dau, output=dataset/output_image_style
python scripts/pipeline_oil_style.py --preset oil

# Tùy chỉnh (sơn dầu)
python scripts/pipeline_oil_style.py dataset/raw/son_dau -o dataset/output_image_style --prefix sondau_style --preset oil

# Preset tranh sơn mài (phong cách Việt)
python scripts/pipeline_oil_style.py dataset/raw/son_dau ^
  --preset sonmai ^
  --style dataset/style_ref/sonmai_1.png
```

Tham số NST (Deep Learning):

- `--style`: ảnh style (mặc định: dataset/style_ref/oil_style.jpg hoặc processed/son_dau)
- `--nst-size`: kích thước ảnh khi chạy NST (512; giảm xuống 256 nếu không có GPU)
- `--nst-steps`: số bước tối ưu (300)
- `--style-weight`, `--content-weight`: trọng số loss
- `--tv-weight`: Total Variation weight (giảm nhiễu/bệt), ví dụ `1e-6`
- `--multiscale`: chạy NST multi-scale (thường ra texture sơn dầu sạch hơn)
- `--nst-scales`: nếu bật multiscale, có thể set `128,256,512` (mặc định auto)
- `--preset`: `oil` (sơn dầu) hoặc `sonmai` (tranh sơn mài, tự set tham số & prefix phù hợp)

Tham số hậu xử lý:

- `--size`: kích thước resize cuối (512)
- `--crop`, `--color-norm`: bật center crop / chuẩn hóa màu
- `--no-flip-h`, `--no-flip-v`, `--no-jitter`: tắt flip / color jitter
- `--max-aug`: số ảnh augmentation tối đa mỗi ảnh gốc (10)

### 3. Chỉ chạy Neural Transfer (một ảnh hoặc thư mục)

Nếu chỉ cần NST không qua pipeline:

```bash
# Một ảnh
python scripts/neural_style_transfer.py path/to/photo.jpg path/to/style.jpg -o output.jpg

# Cả thư mục (cùng một ảnh style)
python scripts/neural_style_transfer.py dataset/raw/son_dau dataset/style_ref/oil_style.jpg -o dataset/out_nst --prefix nst
```

Tham số: `--size`, `--steps`, `--style-weight`, `--content-weight`.

## Preprocess và Augmentation (tùy chọn)

Preprocess (raw → processed):

```bash
python scripts/preprocess.py
```

Augmentation (processed → augmented):

```bash
python scripts/augment.py
```

## Đánh giá FID / LPIPS

So sánh ảnh reference (processed/son_dau) với ảnh sinh ra (output_image_style):

```bash
python scripts/evaluate_fid_lpips.py
```

Tùy chỉnh thư mục:

```bash
python scripts/evaluate_fid_lpips.py --ref dataset/processed/son_dau --gen dataset/output_image_style --pair-by-index
```

Cần: `pip install pytorch-fid lpips` (đã có torch/torchvision).

- **FID**: độ giống phân bố 2 bộ ảnh (thấp = giống hơn).
- **LPIPS**: độ giống perception từng cặp ảnh (thấp = giống hơn).
- `--pair-by-index`: ghép cặp theo thứ tự file khi tên khác nhau.

## Tóm tắt

- **Chỉ tranh sơn dầu**, không dùng Kim Hoàng hay style khác trong repo.
- **Chuyển style 100% bằng Deep Learning**: Neural Style Transfer (Gatys et al., VGG19).
- Pipeline: **raw → NST → resize/crop/color norm → augmentation → output_image_style**.
