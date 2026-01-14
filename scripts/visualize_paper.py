import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
# 1. Đường dẫn dữ liệu đã xử lý (Processed)
PROCESSED_DIR = os.path.join("data", "processed_data")
REID_CLEAN_DIR = os.path.join(PROCESSED_DIR, "reid_merged")
MOT_IMG_DIR = os.path.join(PROCESSED_DIR, "mot_yolo", "images", "train")
MOT_LBL_DIR = os.path.join(PROCESSED_DIR, "mot_yolo", "labels", "train")

# 2. Đường dẫn dữ liệu thô (Raw) - Dùng để đếm so sánh
RAW_DIR = os.path.join("data", "raw_data")
MARKET_RAW = os.path.join(RAW_DIR, "Market-1501-v15.09.15", "bounding_box_train")
DUKE_RAW = os.path.join(RAW_DIR, "DukeMTMC", "bounding_box_train")
MOT_RAW = os.path.join(RAW_DIR, "MOT17", "train")

# 3. Nơi lưu ảnh kết quả
RESULT_DIR = os.path.join("result", "paper_figures")
# ======================================================

def setup_plot_style():
    """Cấu hình font chữ cho đẹp (giống paper)"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rc('font', size=12)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=12)

def count_files(path, ext="*.jpg"):
    if not os.path.exists(path): return 0
    return len(glob.glob(os.path.join(path, ext)))

def get_mot_raw_count():
    # Đếm số frame của các bộ SDP trong raw
    if not os.path.exists(MOT_RAW): return 0
    count = 0
    seqs = [s for s in os.listdir(MOT_RAW) if 'SDP' in s] # Chỉ lấy SDP để so sánh công bằng
    for s in seqs:
        count += count_files(os.path.join(MOT_RAW, s, "img1"))
    return count

def plot_before_after():
    print(">>> 1. Vẽ biểu đồ so sánh Trước/Sau...")
    
    # --- Thu thập số liệu ---
    # Raw
    raw_market = count_files(MARKET_RAW)
    raw_duke = count_files(DUKE_RAW)
    raw_mot = get_mot_raw_count() # Chỉ đếm bộ SDP
    
    # Processed
    # Trong reid_merged, ta cần tách ra đâu là market đâu là duke dựa vào ID offset
    # Nhưng để đơn giản cho biểu đồ, ta đếm tổng hoặc ước lượng
    # Ở đây mình đếm thực tế file trong folder merged
    all_reid_clean = count_files(REID_CLEAN_DIR)
    
    # Để biểu đồ tách biệt, ta giả định tỉ lệ giảm tương đương hoặc lấy số liệu chính xác nếu muốn
    # Vì file đã gộp và đổi tên, ta đếm tổng ReID
    clean_mot = count_files(MOT_IMG_DIR)

    # Dữ liệu vẽ
    categories = ['Market-1501', 'DukeMTMC', 'MOT17 (SDP)']
    
    # Lưu ý: Vì ReID đã gộp, ta tính tổng Raw ReID để so sánh với Tổng Clean ReID
    total_raw_reid = raw_market + raw_duke
    
    # Ve biểu đồ nhóm (ReID Total và MOT17)
    labels = ['ReID Dataset (Merged)', 'Tracking Dataset (MOT17)']
    raw_values = [total_raw_reid, raw_mot]
    clean_values = [all_reid_clean, clean_mot]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, raw_values, width, label='Raw Data (Before)', color='#95a5a6')
    rects2 = ax.bar(x + width/2, clean_values, width, label='Cleaned Data (After)', color='#2ecc71')

    ax.set_ylabel('Số lượng ảnh (Images/Frames)')
    ax.set_title('Thống kê số lượng dữ liệu Trước và Sau khi làm sạch')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Thêm số liệu lên cột
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "Figure_1_Data_Reduction_Stats.png")
    plt.savefig(save_path, dpi=300)
    print(f"   [Saved] {save_path}")
    plt.close()

def visualize_reid_samples():
    print(">>> 2. Tạo ảnh mẫu ReID sau xử lý...")
    files = glob.glob(os.path.join(REID_CLEAN_DIR, "*.jpg"))
    if not files: return
    
    # Lấy ngẫu nhiên 6 ảnh
    samples = random.sample(files, 6)
    
    fig, axes = plt.subplots(1, 6, figsize=(15, 4))
    fig.suptitle("Mẫu dữ liệu ReID sau khi làm sạch (Crop & Resize)", fontsize=16)
    
    for i, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = os.path.basename(img_path)
        
        axes[i].imshow(img)
        axes[i].set_title(f"ID: {name.split('_')[0]}\n{img.shape[0]}x{img.shape[1]}")
        axes[i].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "Figure_2_ReID_Cleaned_Samples.png")
    plt.savefig(save_path, dpi=300)
    print(f"   [Saved] {save_path}")
    plt.close()

def visualize_mot_samples():
    print(">>> 3. Tạo ảnh mẫu Tracking (MOT17 + YOLO Box)...")
    img_files = glob.glob(os.path.join(MOT_IMG_DIR, "*.jpg"))
    if not img_files: return
    
    # Lấy ngẫu nhiên 4 ảnh
    samples = random.sample(img_files, 4)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Mẫu dữ liệu MOT17 và nhãn YOLO sau khi lọc", fontsize=16)
    axes = axes.flatten()
    
    for i, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        h_img, w_img = img.shape[:2]
        
        # Đọc label tương ứng
        lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # YOLO format: class cx cy w h
                    parts = list(map(float, line.strip().split()))
                    # Convert về pixel để vẽ
                    cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
                    
                    x1 = int((cx - w/2) * w_img)
                    y1 = int((cy - h/2) * h_img)
                    x2 = int((cx + w/2) * w_img)
                    y2 = int((cy + h/2) * h_img)
                    
                    # Vẽ box màu xanh lá, dày 2
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(f"Frame: {os.path.basename(img_path)}\nPedestrians: {len(lines)}")
        axes[i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "Figure_3_MOT17_YOLO_Labels.png")
    plt.savefig(save_path, dpi=300)
    print(f"   [Saved] {save_path}")
    plt.close()

def plot_bbox_distribution():
    print(">>> 4. Vẽ biểu đồ phân bố kích thước bbox...")
    lbl_files = glob.glob(os.path.join(MOT_LBL_DIR, "*.txt"))
    # Lấy mẫu khoảng 2000 file để vẽ cho nhanh
    if len(lbl_files) > 2000:
        lbl_files = random.sample(lbl_files, 2000)
        
    heights = []
    
    for lbl in lbl_files:
        with open(lbl, 'r') as f:
            lines = f.readlines()
            for line in lines:
                h = float(line.split()[4]) # Lấy chiều cao (normalized)
                heights.append(h)
                
    if not heights: return

    plt.figure(figsize=(10, 6))
    # Chuyển normalized height sang pixel ước lượng (giả sử ảnh cao TB 1080px)
    # Hoặc để normalized
    plt.hist(heights, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    plt.title('Phân bố chiều cao của người (Normalized Height) sau khi lọc')
    plt.xlabel('Chiều cao (tỷ lệ so với ảnh)')
    plt.ylabel('Số lượng bbox')
    plt.grid(axis='y', alpha=0.5)
    
    save_path = os.path.join(RESULT_DIR, "Figure_4_BBox_Height_Distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"   [Saved] {save_path}")
    plt.close()

if __name__ == "__main__":
    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
    setup_plot_style()
    
    print(f"--- BẮT ĐẦU TRỰC QUAN HÓA (Lưu tại {RESULT_DIR}) ---")
    plot_before_after()
    visualize_reid_samples()
    visualize_mot_samples()
    plot_bbox_distribution()
    print("--- HOÀN TẤT ---")