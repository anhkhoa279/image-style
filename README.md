# Project_Art_Style

Repo này dùng để **tiền xử lý ảnh**, **augmentation**, và tổ chức **dataset theo phong cách hội họa Việt Nam** (Kim Hoàng, Sơn Dầu, Đông Hồ, v.v.), phục vụ nghiên cứu nhận diện/phân loại tranh.

## Cấu trúc thư mục

- **`dataset/raw/<style_name>/`**: ảnh gốc (nhiều định dạng)
- **`dataset/processed/<style_name>/`**: ảnh đã resize, crop (tùy chọn), chuẩn hóa màu (tùy chọn)
- **`dataset/augmented/<style_name>/`**: ảnh tăng cường (flip, rotation, color jitter)
- **`scripts/preprocess.py`**: resize, crop, chuẩn hóa màu
- **`scripts/augment.py`**: Data augmentation (flip, rotation, color jitter)
- **`scripts/evaluate_fid_lpips.py`**: đánh giá FID và LPIPS
- **`docs/VISUAL_INSPECTION.md`**: hướng dẫn kiểm tra chất lượng ảnh bằng mắt
- **`docs/DATASET_SOURCES.md`**: nguồn thu thập dataset 8 loại tranh

Hiện tại bạn đang có dữ liệu ví dụ:

- `dataset/raw/son_dau/` (ảnh gốc)
- `dataset/processed/son_dau/` (ảnh đã xử lý và đặt tên dạng `sondau_001.jpg`…)

## Script `scripts/preprocess.py` làm gì?

Với mỗi folder style (ví dụ `son_dau`):

- Đọc từng file ảnh hợp lệ trong `dataset/raw/<style>/`
- **Resize** về kích thước mục tiêu (mặc định `512x512`)
- Chuẩn hoá ảnh để ghi được **JPG** (xử lý ảnh grayscale / có kênh alpha)
- Lưu sang `dataset/processed/<style>/` với tên theo mẫu: `<prefix>_001.jpg`, `<prefix>_002.jpg`, ...

Trong code đang cấu hình sẵn 2 style:

- `kim_hoang` (nếu chưa có thư mục/ảnh thì script sẽ cảnh báo)
- `son_dau`

## Cách chạy

### 1) Cài môi trường

```bash
python -m pip install -r requirements.txt
```

### 2) Chạy preprocess (từ thư mục gốc project)

```bash
python scripts/preprocess.py
```

## Data Augmentation

```bash
python scripts/augment.py
```

Tạo ảnh tăng cường (flip ngang/dọc, xoay 90/180/270°, color jitter) từ `dataset/processed/` vào `dataset/augmented/`.

## Đánh giá FID / LPIPS

Cần: `pip install pytorch-fid lpips torch torchvision`

```bash
python scripts/evaluate_fid_lpips.py
```

- **FID**: Đo độ giống giữa 2 bộ ảnh (thấp = giống hơn).
- **LPIPS**: Đo độ giống perception giữa các cặp ảnh (thấp = giống hơn).

## Tài liệu bổ sung

- [docs/VISUAL_INSPECTION.md](docs/VISUAL_INSPECTION.md): Kiểm tra chất lượng ảnh theo đặc trưng nghệ thuật
- [docs/DATASET_SOURCES.md](docs/DATASET_SOURCES.md): Nguồn thu thập dataset 8 loại tranh (Artnam, Bảo tàng Mỹ thuật...)

## Thêm style mới

1) Tạo thư mục ảnh gốc: `dataset/raw/<ten_style>/` và bỏ ảnh vào
2) Thêm block gọi `process_images(...)` trong `scripts/preprocess.py` và `augment_dataset(...)` trong `scripts/augment.py`


