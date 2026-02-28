# Project_Art_Style

Repo này dùng để **tiền xử lý ảnh**, **augmentation**, và tổ chức **dataset theo phong cách hội họa Việt Nam** (Kim Hoàng, Sơn Dầu, Đông Hồ, v.v.), phục vụ nghiên cứu nhận diện/phân loại tranh.

## Cấu trúc thư mục

- **`dataset/raw/<style_name>/`**: ảnh gốc (nhiều định dạng)
- **`dataset/processed/<style_name>/`**: ảnh đã resize, crop (tùy chọn), chuẩn hóa màu (tùy chọn)
- **`dataset/augmented/<style_name>/`**: ảnh tăng cường (flip, rotation, color jitter)
- **`dataset/output_image_style/`**: ảnh tranh sơn dầu đã qua pipeline (oil → resize/crop/color norm → augmentation)
- **`dataset/output_kimhoang/`**: ảnh tranh Kim Hoàng đã qua pipeline (kim_hoang → resize/crop/color norm → augmentation)
- **`scripts/preprocess.py`**: resize, crop, chuẩn hóa màu
- **`scripts/augment.py`**: Data augmentation (flip, rotation, color jitter)
- **`scripts/oil_painting_transfer.py`**: chuyển ảnh thường → tranh sơn dầu
- **`scripts/kim_hoang_transfer.py`**: chuyển ảnh thường → tranh Kim Hoàng (nền giấy đỏ, nét phóng khoáng)
- **`scripts/pipeline_oil_style.py`**: pipeline raw → oil → processed → augmented → output_image_style
- **`scripts/pipeline_kim_hoang_style.py`**: pipeline raw → Kim Hoàng → processed → augmented → output_kimhoang
- **`scripts/neural_style_transfer.py`**: Neural Style Transfer (Deep Learning, Gatys et al.)
- **`scripts/evaluate_fid_lpips.py`**: đánh giá FID và LPIPS


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

## Chuyển ảnh sang tranh Kim Hoàng (Kim Hoàng Transfer)

Script `kim_hoang_transfer.py` chuyển ảnh thường sang tranh Kim Hoàng với các đặc điểm:
- **Nền giấy đỏ (giấy điều)**: tranh in trên giấy màu đỏ, còn gọi "tranh Đỏ"
- **Nét vẽ phóng khoáng, tự nhiên, khỏe khoắn, mộc mạc**

```bash
# Một ảnh
python scripts/kim_hoang_transfer.py path/to/photo.jpg -o output_kimhoang.jpg

# Cả thư mục
python scripts/kim_hoang_transfer.py dataset/raw/my_photos/ -o dataset/output_kimhoang/ --prefix kimhoang
```

Tham số: `--red-tint` (0.2–0.5), `--edge-strength`, `--edge-thickness`, `--saturation`, `--warmth`, `--no-texture`.

## Pipeline Kim Hoàng: Raw → Tranh Kim Hoàng → Output_kimhoang

Tương tự pipeline sơn dầu: **raw** → **Kim Hoàng transfer** → **resize, crop, chuẩn hóa màu** → **flip, rotation, color jitter** → `dataset/output_kimhoang/`.

```bash
# Từ thư mục gốc project
python scripts/pipeline_kim_hoang_style.py dataset/raw/kim_hoang -o dataset/output_kimhoang --prefix kimhoang_style

# Mặc định: input=dataset/raw/kim_hoang, output=dataset/output_kimhoang
python scripts/pipeline_kim_hoang_style.py
```

## Neural Style Transfer (Deep Learning)

Script `neural_style_transfer.py` dùng **Neural Transfer** (Gatys et al.) với VGG19 để chuyển phong cách nghệ thuật từ ảnh style sang ảnh content. Tham khảo: [PyTorch Neural Style Tutorial](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html).

Cần: `pip install torch torchvision` (đã có trong requirements.txt)

```bash
# Một ảnh: content + style → output
python scripts/neural_style_transfer.py path/to/photo.jpg path/to/style_painting.jpg -o output_styled.jpg

# Cả thư mục (mỗi ảnh content dùng chung 1 ảnh style)
python scripts/neural_style_transfer.py dataset/raw/son_dau dataset/processed/son_dau/sondau_001.jpg -o dataset/output_neural_style --prefix nst_oil
```

Tham số: `--size` (512 mặc định, giảm nếu không có GPU), `--steps` (300), `--style-weight` (1e6), `--content-weight` (1).

## Đánh giá FID / LPIPS

Cần: `pip install pytorch-fid lpips torch torchvision`

```bash
# Sơn dầu (mặc định)
python scripts/evaluate_fid_lpips.py

# Kim Hoàng
python scripts/evaluate_fid_lpips.py --style kim_hoang

# Tùy chỉnh thư mục
python scripts/evaluate_fid_lpips.py --ref dataset/processed/kim_hoang --gen dataset/output_kimhoang --pair-by-index
```

- **FID**: Đo độ giống giữa 2 bộ ảnh (thấp = giống hơn).
- **LPIPS**: Đo độ giống perception giữa các cặp ảnh (thấp = giống hơn).
- `--pair-by-index`: Ghép cặp theo thứ tự file khi tên khác nhau.