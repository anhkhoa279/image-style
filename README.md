# Project_Art_Style

Repo này dùng để **tiền xử lý ảnh**, **augmentation**, và tổ chức **dataset theo phong cách hội họa Việt Nam** (Kim Hoàng, Sơn Dầu, Đông Hồ, v.v.), phục vụ nghiên cứu nhận diện/phân loại tranh.

## Cấu trúc thư mục

- **`dataset/raw/<style_name>/`**: ảnh gốc (nhiều định dạng)
- **`dataset/processed/<style_name>/`**: ảnh đã resize, crop (tùy chọn), chuẩn hóa màu (tùy chọn)
- **`dataset/augmented/<style_name>/`**: ảnh tăng cường (flip, rotation, color jitter)
- **`dataset/output_image_style/`**: ảnh tranh sơn dầu đã qua pipeline (oil → resize/crop/color norm → augmentation), một folder đồng nhất
- **`scripts/preprocess.py`**: resize, crop, chuẩn hóa màu
- **`scripts/augment.py`**: Data augmentation (flip, rotation, color jitter)
- **`scripts/oil_painting_transfer.py`**: chuyển ảnh thường → tranh sơn dầu
- **`scripts/pipeline_oil_style.py`**: pipeline đồng nhất raw → oil → processed → augmented → output_image_style
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

## Chuyển ảnh sang tranh sơn dầu (Oil Painting Transfer)

Script `oil_painting_transfer.py` chuyển ảnh thường sang tranh sơn dầu với các đặc điểm:
- Chất liệu sơn dầu, mảng khối 3D (nét cọ rõ, impasto)
- Tông màu ấm áp / hoài niệm, ánh sáng tinh tế
- Kết cấu vải canvas, hiệu ứng chiều sâu

```bash
# Một ảnh
python scripts/oil_painting_transfer.py path/to/photo.jpg -o output_oil.jpg

# Cả thư mục
python scripts/oil_painting_transfer.py dataset/raw/my_photos/ -o dataset/output_oil/ --prefix sondau
```

Tham số: `--brush-size` (5–8), `--oil-blend` (0.4–0.55), `--impasto` (0.1–0.18), `--warmth`, `--vignette`, `--no-texture`.

## Pipeline đồng nhất: Raw → Tranh sơn dầu → Output_image_style

Script `pipeline_oil_style.py` gộp toàn bộ vào một luồng: **raw** → **oil painting** → **resize, crop, chuẩn hóa màu** → **flip, rotation, color jitter** → lưu vào **một folder** `dataset/output_image_style/`.

```bash
# Từ thư mục gốc project (có dataset/raw/...)
python scripts/pipeline_oil_style.py dataset/raw/son_dau -o dataset/output_image_style --prefix sondau_style

# Mặc định: input=dataset/raw/son_dau, output=dataset/output_image_style
python scripts/pipeline_oil_style.py
```

Tùy chọn: `--size 512`, `--crop`, `--color-norm`, `--no-flip-h`, `--no-flip-v`, `--no-jitter`, `--max-aug 10`.

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


