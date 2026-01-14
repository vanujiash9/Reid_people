# Hệ thống Định danh và Theo dõi Đa đối tượng

**YOLOv8 + ByteTrack + ResNet50 ReID**

**Tác giả:** Thanh Vân  
**Email:** thanh.van19062004@gmail.com  
**Ngày:** Tháng 1, 2026

---

## 1. Tổng quan

Hệ thống phát hiện, theo dõi và định danh người trong video với ID toàn cục ổn định.

### Pipeline

```
Video → YOLOv8 → ByteTrack → ReID Embedding → Cosine Matching → Global ID
```

---

## 2. Kiến trúc Tracker (DeepSORT-style)

### 2.1 Máy trạng thái Track

**Tentative (thử nghiệm):** hits < 3, ưu tiên IoU  
**Confirmed (đã xác nhận):** hits ≥ 3, ưu tiên ReID  
**Deleted (đã xóa):** time_since_update > 30 frames

### 2.2 Ma trận Chi phí

**Track Tentative:**
- Cost = 0.7 × IoU_distance + 0.3 × Cosine_distance

**Track Confirmed có overlap (IoU > 0.3):**
- Cost = 0.3 × IoU_distance + 0.7 × Cosine_distance

**Track Confirmed không overlap:**
- Cost = Cosine_distance (100%)

### 2.3 Ngưỡng

- Max Cosine Distance: 0.4
- Max IoU Distance: 0.7
- Max Age: 30 frames
- N_Init: 3 frames
- Min Confidence: 0.5

### 2.4 Hungarian Algorithm

Sử dụng linear_sum_assignment tìm cặp tối ưu tracks-detections.

---

## 3. Dữ liệu

### 3.1 ReID: Market-1501 + DukeMTMC

**Làm sạch:**
- Loại ảnh nhiễu: pid = -1 hoặc 0000
- Lọc kích thước: min < 32px
- Remap ID: new_pid = old_pid + offset

**Phân chia:**
- Train: 26,512 ảnh
- Validation: 2,946 ảnh  
- Query: 3,368 ảnh
- Gallery: 19,732 ảnh

### 3.2 Detection: MOT17

**Làm sạch:**
- Chỉ pedestrian (class = 1)
- Visibility ≥ 0.2
- Bbox width ≥ 10px
- Chỉ dùng SDP sequence

**Phân chia:**
- Train/Val: 5,316 frames

---

## 4. Mô hình ReID

### Kiến trúc
- Backbone: ResNet50 pretrained ImageNet
- Last stride = 1 (giữ độ phân giải cao)
- Input: 256×128

### Loss Functions
- Cross-Entropy: phân loại 1,453 IDs
- Triplet Loss: anchor-positive-negative

### Quản lý Features
- Exponential Moving Average (ưu tiên features gần)
- Max 50 features/track
- L2 normalization

---

## 5. Tối ưu hóa

### ReID Extraction
- Skip frames: extract mỗi 3 frames
- Batch size: 8 crops/lần
- Feature update: mỗi 10 frames

### Video Processing
- Codec ưu tiên: H264/AVC1
- Fallback: MP4V, XVID
- Auto re-encode với ffmpeg
- Tắt display để tăng tốc

### Lưu Crops
- Interval: mỗi 10 frames/track
- Min quality: 0.5
- Tổ chức: ID_XXX/frame_XXXXXX_conf_X.XX.jpg

---

## 6. Đánh giá

### Metrics
- Rank-1 Accuracy
- mAP (mean Average Precision)
- Cosine Distance Distribution

### Output Real-time
- Total IDs: tổng người duy nhất
- Active Tracks: đang theo dõi
- Confirmed: đã xác thực

---

## 7. Cấu trúc Output

```
output/
├── tracked_video_final.mp4
├── tracking_metadata.json
└── person_crops/
    ├── ID_001/
    └── ID_002/
```

### Metadata JSON
- Video info: resolution, fps, thời gian xử lý
- Tracking summary: tổng người, config
- Per-person: first/last frame, số frames, crops đã lưu

---

## 8. Ứng dụng

- Giám sát an ninh
- Phân tích đám đông
- Retail analytics
- Nghiên cứu MOT & ReID
