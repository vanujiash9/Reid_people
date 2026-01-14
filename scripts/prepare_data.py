import os
import shutil
import glob
import pandas as pd
import configparser
from tqdm import tqdm
import cv2

# ================= CẤU HÌNH (ĐÃ CHỈNH THEO MÁY BẠN) =================
# Đường dẫn gốc chứa dữ liệu thô
RAW_DATA_DIR = os.path.join("data", "raw_data")

# Đường dẫn dataset con
MARKET_DIR = os.path.join(RAW_DATA_DIR, "Market-1501-v15.09.15", "bounding_box_train")
DUKE_DIR = os.path.join(RAW_DATA_DIR, "DukeMTMC", "bounding_box_train") # Đã sửa tên folder
MOT17_DIR = os.path.join(RAW_DATA_DIR, "MOT17", "train")

# Đường dẫn đầu ra (Nơi chứa dữ liệu sạch để train)
OUTPUT_DIR = os.path.join("data", "processed_data")
REID_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "reid_merged")
YOLO_IMG_DIR = os.path.join(OUTPUT_DIR, "mot_yolo", "images", "train")
YOLO_LABEL_DIR = os.path.join(OUTPUT_DIR, "mot_yolo", "labels", "train")

# Các thông số lọc
MIN_IMG_SIZE = 32          # ReID: Bỏ ảnh nhỏ hơn 32px
MOT_CONF_THRES = 0.2       # MOT: Chỉ lấy bbox rõ ràng
MOT_MIN_HEIGHT = 10        # MOT: Bỏ bbox quá bé
# ====================================================================

def create_dirs():
    """Tạo cấu trúc thư mục đầu ra"""
    for d in [REID_OUTPUT_DIR, YOLO_IMG_DIR, YOLO_LABEL_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
    print(f"[INFO] Đã tạo thư mục output tại: {OUTPUT_DIR}")

def process_reid():
    print("\n>>> 1. XỬ LÝ REID (Market + Duke)...")
    
    # --- Check folder tồn tại ---
    if not os.path.exists(MARKET_DIR) or not os.path.exists(DUKE_DIR):
        print("[LỖI] Không tìm thấy thư mục Market hoặc Duke. Kiểm tra lại đường dẫn!")
        return

    # --- Bước 1: Market-1501 ---
    market_files = glob.glob(os.path.join(MARKET_DIR, "*.jpg"))
    max_market_id = 0
    count_market = 0

    print(f"   - Đang copy Market-1501 ({len(market_files)} ảnh)...")
    for file_path in tqdm(market_files, desc="Processing Market"):
        filename = os.path.basename(file_path)
        pid_str = filename.split('_')[0]
        
        # Lọc ID rác (-1 và 0000)
        if pid_str == "0000" or pid_str == "-1": continue
        
        # Check size ảnh nhanh
        img = cv2.imread(file_path)
        if img is None or min(img.shape[:2]) < MIN_IMG_SIZE: continue

        pid = int(pid_str)
        if pid > max_market_id: max_market_id = pid
        
        shutil.copy(file_path, os.path.join(REID_OUTPUT_DIR, filename))
        count_market += 1

    # --- Bước 2: DukeMTMC (Cộng Offset ID) ---
    duke_files = glob.glob(os.path.join(DUKE_DIR, "*.jpg"))
    count_duke = 0
    id_offset = max_market_id + 100 
    
    print(f"   - Đang gộp DukeMTMC ({len(duke_files)} ảnh). ID Offset bắt đầu từ: {id_offset}")
    
    for file_path in tqdm(duke_files, desc="Processing Duke"):
        filename = os.path.basename(file_path)
        pid_str = filename.split('_')[0]
        
        if not pid_str.isdigit(): continue
        
        # Đổi ID cũ -> ID mới
        old_pid = int(pid_str)
        new_pid = old_pid + id_offset
        
        # Tạo tên file mới chuẩn format: 01502_c1s1_...
        parts = filename.split('_')
        parts[0] = f"{new_pid:05d}"
        new_filename = "_".join(parts)
        
        img = cv2.imread(file_path)
        if img is None or min(img.shape[:2]) < MIN_IMG_SIZE: continue

        shutil.copy(file_path, os.path.join(REID_OUTPUT_DIR, new_filename))
        count_duke += 1

    print(f"-> XONG REID. Tổng ảnh sạch: {count_market + count_duke}")

def convert_box(size, box):
    """Tính toán toạ độ cho YOLO (Normalized center_x, center_y, w, h)"""
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[0] + box[2])/2.0
    y = (box[1] + box[1] + box[3])/2.0
    w = box[2]
    h = box[3]
    return (x*dw, y*dh, w*dw, h*dh)

def process_mot():
    print("\n>>> 2. XỬ LÝ MOT17 (Convert sang YOLO)...")
    if not os.path.exists(MOT17_DIR):
        print("[LỖI] Không tìm thấy thư mục MOT17.")
        return

    # Chỉ lấy các folder SDP để tránh trùng lặp dữ liệu (Vì DPM, FRCNN, SDP chung 1 video)
    # Nếu bạn muốn nhiều data hơn thì bỏ dòng if 'SDP' bên dưới
    seqs = [s for s in os.listdir(MOT17_DIR) if 'SDP' in s] 
    
    print(f"   - Tìm thấy {len(seqs)} chuỗi video (Chỉ lấy bộ SDP để tránh trùng lặp).")
    total_frames = 0

    for seq in tqdm(seqs, desc="Converting MOT17"):
        seq_path = os.path.join(MOT17_DIR, seq)
        img_dir = os.path.join(seq_path, "img1")
        gt_path = os.path.join(seq_path, "gt", "gt.txt")
        ini_path = os.path.join(seq_path, "seqinfo.ini")
        
        if not os.path.exists(gt_path): continue

        config = configparser.ConfigParser()
        config.read(ini_path)
        img_w = int(config['Sequence']['imWidth'])
        img_h = int(config['Sequence']['imHeight'])

        # Đọc file GT
        df = pd.read_csv(gt_path, header=None)
        
        # Lọc: Class=1 (Người đi bộ), Visibility >= 0.2
        # Cột: 0=frame, 1=id, 2=left, 3=top, 4=width, 5=height, ... 7=class, 8=vis
        df = df[df[7] == 1] 
        df = df[df[8] >= MOT_CONF_THRES]
        df = df[df[5] >= MOT_MIN_HEIGHT]

        grouped = df.groupby(0) # Nhóm theo Frame ID

        images = glob.glob(os.path.join(img_dir, "*.jpg"))
        for img_path in images:
            base_name = os.path.basename(img_path)
            frame_id = int(base_name.split('.')[0])
            
            # Đổi tên file để không trùng giữa các video
            new_name = f"{seq}_{base_name}"
            
            # Copy ảnh sang folder YOLO
            shutil.copy(img_path, os.path.join(YOLO_IMG_DIR, new_name))
            
            # Tạo file label txt
            label_file = new_name.replace(".jpg", ".txt")
            label_path = os.path.join(YOLO_LABEL_DIR, label_file)
            
            content = []
            if frame_id in grouped.groups:
                data = grouped.get_group(frame_id)
                for _, row in data.iterrows():
                    bbox = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                    bb = convert_box((img_w, img_h), bbox)
                    # Class 0 cho person
                    content.append(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(content))
            
            total_frames += 1

    print(f"-> XONG MOT17. Đã tạo {total_frames} ảnh/label cho YOLO.")

if __name__ == "__main__":
    create_dirs()
    process_reid()
    process_mot()
    print("\n========================================")
    print("HOÀN TẤT! Dữ liệu nằm tại 'data/processed_data'")