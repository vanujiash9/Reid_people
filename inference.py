import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import deque, defaultdict
from pathlib import Path
import json
from datetime import datetime
from src.models.reid_model import ReIDResNet50

# ================= CONFIG =================
VIDEO_PATH = "videoplayback (1).mp4"
YOLO_WEIGHTS = "weights/yolov8_mot17_results/weights/best.pt"
REID_WEIGHTS = "weights/reid_best.pth"

# Output paths
OUTPUT_DIR = Path("output")
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "tracked_video.mp4"
CROPS_DIR = OUTPUT_DIR / "person_crops"
METADATA_PATH = OUTPUT_DIR / "tracking_metadata.json"

# Tracking config - Inspired by DeepSORT/StrongSORT
MAX_COSINE_DISTANCE = 0.4   # Cosine distance threshold cho ReID
MAX_IOU_DISTANCE = 0.7      # IoU distance threshold
MAX_AGE = 30                # Số frame tối đa để giữ track khi mất
N_INIT = 3                  # Số frame khởi tạo để confirm track
MIN_CONFIDENCE = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Feature config
MAX_FEATURES_PER_TRACK = 50
FEATURE_UPDATE_INTERVAL = 10  # Cập nhật ít hơn = nhanh hơn

# Performance optimization
SKIP_REID_FRAMES = 3  # Chỉ extract ReID mỗi 3 frames
BATCH_SIZE = 8        # Extract nhiều crops cùng lúc

# Crop saving config
SAVE_CROPS = True
CROP_SAVE_INTERVAL = 10
MIN_CROP_QUALITY = 0.5
# ==========================================

def iou(bbox1, bbox2):
    """Tính IoU giữa 2 bounding boxes"""
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_ - x1_) * (y2_ - y1_)
    
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

class Track:
    """Track object - inspired by DeepSORT"""
    def __init__(self, track_id, bbox, feature, frame_idx):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.features = deque([feature], maxlen=MAX_FEATURES_PER_TRACK)
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.state = 'tentative'  # tentative or confirmed
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.crop_count = 0
        
    def update(self, bbox, feature, frame_idx):
        """Cập nhật track với detection mới"""
        self.bbox = bbox
        if feature is not None:
            self.features.append(feature)
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        
        # Chuyển sang confirmed sau N_INIT hits
        if self.state == 'tentative' and self.hits >= N_INIT:
            self.state = 'confirmed'
    
    def mark_missed(self):
        """Đánh dấu track bị miss"""
        self.hit_streak = 0
        self.time_since_update += 1
    
    def is_tentative(self):
        return self.state == 'tentative'
    
    def is_confirmed(self):
        return self.state == 'confirmed'
    
    def is_deleted(self):
        return self.time_since_update > MAX_AGE
    
    def get_feature(self):
        """Lấy feature trung bình của track"""
        features = np.array(list(self.features))
        # Sử dụng EMA (Exponential Moving Average) để ưu tiên features gần đây
        weights = np.exp(np.linspace(-1, 0, len(features)))
        weights /= weights.sum()
        avg_feature = np.average(features, axis=0, weights=weights)
        return avg_feature / (np.linalg.norm(avg_feature) + 1e-8)

class Tracker:
    """Multi-object tracker - inspired by DeepSORT"""
    def __init__(self):
        self.tracks = []
        self.next_id = 0
        self.frame_idx = 0
        
    def update(self, detections, features):
        """
        Cập nhật tracker với detections mới
        detections: List of [x1, y1, x2, y2, confidence]
        features: List of ReID features (hoặc None)
        """
        self.frame_idx += 1
        
        # Bước 1: Dự đoán vị trí mới của tracks (simplified - chỉ giữ nguyên)
        for track in self.tracks:
            track.age += 1
        
        # Bước 2: Tính cost matrix (kết hợp IoU + ReID)
        if len(self.tracks) == 0:
            # Không có track nào -> tạo tracks mới
            for det, feat in zip(detections, features):
                if feat is not None:
                    self._initiate_track(det, feat)
            return self.tracks
        
        if len(detections) == 0:
            # Không có detection -> đánh dấu tất cả tracks bị miss
            for track in self.tracks:
                track.mark_missed()
            self._cleanup_tracks()
            return self.tracks
        
        # Tính cost matrix
        cost_matrix = self._compute_cost_matrix(detections, features)
        
        # Bước 3: Hungarian algorithm để matching
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Bước 4: Xử lý matches
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] < 1.0:  # Valid match
                matches.append((t_idx, d_idx))
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)
                if d_idx in unmatched_detections:
                    unmatched_detections.remove(d_idx)
        
        # Cập nhật matched tracks
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(
                detections[d_idx][:4],
                features[d_idx],
                self.frame_idx
            )
        
        # Đánh dấu unmatched tracks
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()
        
        # Tạo tracks mới cho unmatched detections
        for d_idx in unmatched_detections:
            if features[d_idx] is not None:
                self._initiate_track(detections[d_idx], features[d_idx])
        
        # Cleanup tracks cũ
        self._cleanup_tracks()
        
        return self.tracks
    
    def _compute_cost_matrix(self, detections, features):
        """Tính cost matrix kết hợp IoU và ReID"""
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                # 1. IoU distance
                iou_score = iou(track.bbox, det[:4])
                iou_cost = 1.0 - iou_score
                
                # 2. ReID distance (cosine)
                if features[d_idx] is not None:
                    track_feature = track.get_feature()
                    det_feature = features[d_idx]
                    cosine_dist = cdist([track_feature], [det_feature], metric='cosine')[0][0]
                else:
                    cosine_dist = MAX_COSINE_DISTANCE
                
                # 3. Kết hợp cost
                # Ưu tiên IoU cho tracks mới (tentative)
                # Ưu tiên ReID cho tracks confirmed
                if track.is_tentative():
                    # Track mới -> tin IoU nhiều hơn
                    cost = 0.7 * iou_cost + 0.3 * cosine_dist
                else:
                    # Track confirmed -> tin ReID nhiều hơn
                    if iou_score > 0.3:  # Có overlap
                        cost = 0.3 * iou_cost + 0.7 * cosine_dist
                    else:
                        # Không overlap -> chỉ dùng ReID
                        cost = cosine_dist
                
                # Gating: reject nếu cost quá cao
                if cosine_dist > MAX_COSINE_DISTANCE or iou_cost > MAX_IOU_DISTANCE:
                    cost = 1e5  # Infinite cost
                
                cost_matrix[t_idx, d_idx] = cost
        
        return cost_matrix
    
    def _initiate_track(self, detection, feature):
        """Tạo track mới"""
        track = Track(self.next_id, detection[:4], feature, self.frame_idx)
        self.tracks.append(track)
        self.next_id += 1
    
    def _cleanup_tracks(self):
        """Xóa tracks cũ"""
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
    
    def get_confirmed_tracks(self):
        """Lấy các tracks đã confirmed"""
        return [t for t in self.tracks if t.is_confirmed()]

def extract_embedding(model, crop, tfm):
    """Extract ReID embedding"""
    if crop.size == 0:
        return None
    
    try:
        h, w = crop.shape[:2]
        if h < 50 or w < 30:
            return None
        
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        tensor = tfm(crop_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            feat = model(tensor).cpu().numpy()[0]
            feat /= (np.linalg.norm(feat) + 1e-8)
        
        return feat
    except:
        return None

def extract_embeddings_batch(model, crops, tfm):
    """Extract ReID embeddings cho nhiều crops cùng lúc - NHANH HƠN"""
    if len(crops) == 0:
        return []
    
    valid_crops = []
    valid_indices = []
    
    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            continue
        h, w = crop.shape[:2]
        if h >= 50 and w >= 30:
            valid_crops.append(crop)
            valid_indices.append(i)
    
    if len(valid_crops) == 0:
        return [None] * len(crops)
    
    try:
        # Convert all crops to tensors
        tensors = []
        for crop in valid_crops:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            tensor = tfm(crop_pil)
            tensors.append(tensor)
        
        # Batch processing
        batch = torch.stack(tensors).to(DEVICE)
        
        with torch.no_grad():
            features = model(batch).cpu().numpy()
            # Normalize
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Map back to original indices
        results = [None] * len(crops)
        for i, feat in zip(valid_indices, features):
            results[i] = feat
        
        return results
    except:
        return [None] * len(crops)

def get_color_for_id(track_id):
    """Màu cố định cho mỗi ID"""
    np.random.seed(track_id * 789)
    color = tuple(np.random.randint(100, 255, 3).tolist())
    return color

def save_person_crop(crop, track_id, frame_idx, confidence):
    """Lưu crop của người vào thư mục riêng"""
    person_dir = CROPS_DIR / f"ID_{track_id:03d}"
    person_dir.mkdir(parents=True, exist_ok=True)
    
    crop_filename = person_dir / f"frame_{frame_idx:06d}_conf_{confidence:.2f}.jpg"
    cv2.imwrite(str(crop_filename), crop)
    
    return crop_filename

def main():
    # Tạo thư mục output
    OUTPUT_DIR.mkdir(exist_ok=True)
    CROPS_DIR.mkdir(exist_ok=True)
    
    print("[INFO] Loading models...")
    detector = YOLO(YOLO_WEIGHTS)
    
    reid_model = ReIDResNet50(num_classes=1453)
    reid_model.load_state_dict(torch.load(REID_WEIGHTS, map_location=DEVICE))
    reid_model.to(DEVICE).eval()
    
    tracker = Tracker()
    update_counter = defaultdict(int)
    
    tfm = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer với codec tốt hơn
    output_video_path = OUTPUT_VIDEO_PATH
    
    # Thử H264 trước, fallback về mp4v nếu không có
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 - tương thích tốt nhất
    except:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback
    
    out = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )
    
    if not out.isOpened():
        print("[ERROR] Cannot create video writer! Trying alternative codec...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video_path = OUTPUT_DIR / "tracked_video.avi"
        out = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (frame_width, frame_height)
        )
    
    print(f"[INFO] DeepSORT-style Tracking System")
    print(f"[INFO] Video: {VIDEO_PATH}")
    print(f"[INFO] Output: {output_video_path}")
    print(f"[CONFIG] Max cosine distance: {MAX_COSINE_DISTANCE}")
    print(f"[CONFIG] Max IoU distance: {MAX_IOU_DISTANCE}")
    print(f"[CONFIG] Max age: {MAX_AGE} frames")
    print(f"[CONFIG] N_init: {N_INIT} frames")
    print(f"[CONFIG] Skip ReID frames: {SKIP_REID_FRAMES}")
    print("=" * 80)
    
    start_time = datetime.now()
    last_reid_frame = 0  # Track khi nào extract ReID
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Progress
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100
            confirmed = len(tracker.get_confirmed_tracks())
            elapsed = (datetime.now() - start_time).total_seconds()
            fps_processing = frame_idx / elapsed if elapsed > 0 else 0
            print(f"[PROGRESS] Frame {frame_idx}/{total_frames} ({progress:.1f}%) | "
                  f"Tracks: {len(tracker.tracks)} | Confirmed: {confirmed} | "
                  f"Speed: {fps_processing:.1f} fps")
        
        # Detection
        results = detector(frame, verbose=False, conf=MIN_CONFIDENCE)
        
        if len(results[0].boxes) == 0:
            tracker.update([], [])
            out.write(frame)
            # Tắt hiển thị để chạy nhanh hơn
            # cv2.imshow("DeepSORT Tracking", frame)
            # if cv2.waitKey(1) == 27:
            #     break
            continue
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Chỉ extract ReID features mỗi SKIP_REID_FRAMES frames
        should_extract_reid = (frame_idx - last_reid_frame) >= SKIP_REID_FRAMES
        
        detections = []
        crops_list = []
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)
            
            detections.append([x1, y1, x2, y2, conf])
            
            if should_extract_reid:
                crop = frame[y1:y2, x1:x2]
                crops_list.append(crop)
            else:
                crops_list.append(None)
        
        # Extract features (batch nếu cần)
        if should_extract_reid:
            features = extract_embeddings_batch(reid_model, crops_list, tfm)
            last_reid_frame = frame_idx
        else:
            features = [None] * len(detections)
        
        # Update tracker
        tracks = tracker.update(detections, features)
        
        # Vẽ và lưu crops
        for track in tracker.get_confirmed_tracks():
            x1, y1, x2, y2 = map(int, track.bbox)
            track_id = track.track_id
            
            # Lưu crop định kỳ
            if SAVE_CROPS:
                update_counter[track_id] += 1
                if update_counter[track_id] % CROP_SAVE_INTERVAL == 0:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        save_person_crop(crop, track_id, frame_idx, 0.9)
                        track.crop_count += 1
            
            # Vẽ
            color = get_color_for_id(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID{track_id}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # UI overlay
        confirmed_tracks = tracker.get_confirmed_tracks()
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 40
        cv2.putText(frame, f"Total IDs: {tracker.next_id}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Active Tracks: {len(tracker.tracks)}", 
                   (20, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confirmed: {len(confirmed_tracks)}", 
                   (20, y+65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", 
                   (20, y+95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        out.write(frame)
        
        # Tắt hiển thị real-time để chạy nhanh hơn
        # Bỏ comment 2 dòng dưới nếu muốn xem real-time (sẽ chậm hơn)
        # cv2.imshow("DeepSORT Tracking", frame)
        # if cv2.waitKey(1) == 27:
        #     break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n[INFO] Re-encoding video with ffmpeg for better compatibility...")
    
    # Re-encode video bằng ffmpeg (nếu có)
    try:
        import subprocess
        
        temp_video = output_video_path
        final_video = OUTPUT_DIR / "tracked_video_final.mp4"
        
        # Lệnh ffmpeg để encode lại với H264
        cmd = [
            'ffmpeg',
            '-i', str(temp_video),
            '-c:v', 'libx264',        # H264 codec
            '-preset', 'medium',       # Cân bằng tốc độ/chất lượng
            '-crf', '23',             # Chất lượng (18-28 là tốt)
            '-pix_fmt', 'yuv420p',    # Tương thích với hầu hết players
            '-y',                      # Overwrite
            str(final_video)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[SUCCESS] Video re-encoded: {final_video}")
            # Xóa video tạm
            temp_video.unlink()
            output_video_path = final_video
        else:
            print(f"[WARNING] ffmpeg failed. Using original video.")
            print(f"Install ffmpeg: https://ffmpeg.org/download.html")
    except FileNotFoundError:
        print("[INFO] ffmpeg not found. Video saved in original format.")
        print("For better compatibility, install ffmpeg and run:")
        print(f"  ffmpeg -i {output_video_path} -c:v libx264 -crf 23 output_final.mp4")
    except Exception as e:
        print(f"[WARNING] Re-encoding failed: {e}")
    
    # Lưu metadata
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    metadata = {
        'video_info': {
            'input_path': VIDEO_PATH,
            'output_path': str(output_video_path),
            'resolution': f"{frame_width}x{frame_height}",
            'fps': fps,
            'total_frames': frame_idx,
            'processing_time_seconds': processing_time
        },
        'tracking_summary': {
            'total_unique_persons': tracker.next_id,
            'algorithm': 'DeepSORT-style',
            'config': {
                'max_cosine_distance': MAX_COSINE_DISTANCE,
                'max_iou_distance': MAX_IOU_DISTANCE,
                'max_age': MAX_AGE,
                'n_init': N_INIT
            }
        },
        'persons': []
    }
    
    # Thêm thông tin tracks
    for track in tracker.tracks:
        if track.is_confirmed():
            metadata['persons'].append({
                'id': track.track_id,
                'first_frame': track.first_frame,
                'last_frame': track.last_frame,
                'total_frames': track.hits,
                'crops_saved': track.crop_count
            })
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"[COMPLETE] Processing finished!")
    print(f"[SUMMARY]")
    print(f"  • Total unique persons: {tracker.next_id}")
    print(f"  • Processing time: {processing_time:.1f}s ({frame_idx/processing_time:.1f} fps)")
    print(f"\n[OUTPUT]")
    print(f"  • Video: {output_video_path}")
    print(f"  • Crops: {CROPS_DIR}")
    print(f"  • Metadata: {METADATA_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()