import os
import shutil
import glob
import pandas as pd
import configparser
from tqdm import tqdm

# ================= CẤU HÌNH =================
RAW_DATA_DIR = os.path.join("data", "raw_data")
MOT17_DIR = os.path.join(RAW_DATA_DIR, "MOT17", "train")

# Chỉ tác động vào folder mot_yolo
OUTPUT_DIR = os.path.join("data", "processed_data", "mot_yolo")
YOLO_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "train")
YOLO_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels", "train")

# Thông số lọc
MOT_CONF_THRES = 0.2
MOT_MIN_HEIGHT = 10
# ============================================

def convert_box_safe(size, box):
    """
    Convert (x, y, w, h) sang (cx, cy, nw, nh) 
    Và ép các con số vào khoảng [0.0, 1.0] để tránh lỗi Corrupt
    """
    img_w, img_h = size
    left, top, width, height = box
    
    # 1. Tính tâm và kích thước chuẩn hóa
    cx = (left + width / 2.0) / img_w
    cy = (top + height / 2.0) / img_h
    nw = width / img_w
    nh = height / img_h
    
    # 2. GHÌM TỌA ĐỘ (QUAN TRỌNG): 
    # Nếu người đứng ngoài khung hình, ta ép box về sát mép thay vì để tọa độ âm
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    
    return (cx, cy, nw, nh)

def clean_yolo_data():
    # 1. Xóa folder cũ nếu có để làm lại từ đầu cho sạch
    if os.path.exists(OUTPUT_DIR):
        print(f">>> Đang xóa dữ liệu cũ tại {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(YOLO_IMG_DIR, exist_ok=True)
    os.makedirs(YOLO_LABEL_DIR, exist_ok=True)

    if not os.path.exists(MOT17_DIR):
        print("[LỖI] Không tìm thấy MOT17.")
        return

    # Chỉ lấy bộ SDP để tránh trùng lặp video
    seqs = [s for s in os.listdir(MOT17_DIR) if 'SDP' in s]
    total_frames = 0

    print("\n>>> BẮT ĐẦU LÀM SẠCH LẠI DỮ LIỆU MOT-YOLO...")
    for seq in tqdm(seqs, desc="Processing Sequences"):
        seq_path = os.path.join(MOT17_DIR, seq)
        img_dir = os.path.join(seq_path, "img1")
        gt_path = os.path.join(seq_path, "gt", "gt.txt")
        ini_path = os.path.join(seq_path, "seqinfo.ini")
        
        if not os.path.exists(gt_path): continue

        # Đọc kích thước ảnh
        config = configparser.ConfigParser()
        config.read(ini_path)
        img_w = int(config['Sequence']['imWidth'])
        img_h = int(config['Sequence']['imHeight'])

        # Đọc nhãn
        df = pd.read_csv(gt_path, header=None)
        
        # Lọc Class=1 (Người đi bộ), Vis > 0.2
        df = df[df[7] == 1] 
        df = df[df[8] >= MOT_CONF_THRES]
        df = df[df[5] >= MOT_MIN_HEIGHT]

        grouped = df.groupby(0)

        images = glob.glob(os.path.join(img_dir, "*.jpg"))
        for img_path in images:
            base_name = os.path.basename(img_path)
            frame_id = int(base_name.split('.')[0])
            new_name = f"{seq}_{base_name}"
            
            # Copy ảnh
            shutil.copy(img_path, os.path.join(YOLO_IMG_DIR, new_name))
            
            # Tạo nhãn chuẩn hóa
            label_file = new_name.replace(".jpg", ".txt")
            content = []
            if frame_id in grouped.groups:
                data = grouped.get_group(frame_id)
                for _, row in data.iterrows():
                    # box: left(2), top(3), width(4), height(5)
                    bbox_raw = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                    
                    # Dùng hàm safe convert
                    cx, cy, nw, nh = convert_box_safe((img_w, img_h), bbox_raw)
                    
                    # Chỉ lưu nếu box có kích thước hợp lệ sau khi ép
                    if nw > 0 and nh > 0:
                        content.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
            with open(os.path.join(YOLO_LABEL_DIR, label_file), 'w') as f:
                f.write('\n'.join(content))
            
            total_frames += 1

    print(f"\n[XONG] Đã làm sạch xong {total_frames} ảnh.")
    print(f"Dữ liệu sẵn sàng tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    clean_yolo_data()