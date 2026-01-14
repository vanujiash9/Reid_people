import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# ================= CẤU HÌNH (ĐÃ CHỈNH THEO ẢNH CỦA BẠN) =================
# Dựa trên ảnh: Folder gốc là "data/raw_data"
BASE_DIR = os.path.join("data", "raw_data")

# Tên folder con (Khớp chính xác với ảnh chụp màn hình)
MARKET_DIR = os.path.join(BASE_DIR, "Market-1501-v15.09.15")
DUKE_DIR = os.path.join(BASE_DIR, "DukeMTMC")  # Đã sửa từ DukeMTMC-reID thành DukeMTMC
MOT_DIR = os.path.join(BASE_DIR, "MOT17", "train")

# Thư mục lưu kết quả
RESULT_DIR = "result"
# ======================================================================

def count_files(directory, ext="*.jpg"):
    """Đếm số file trong thư mục"""
    if not os.path.exists(directory):
        return 0
    return len(glob.glob(os.path.join(directory, ext)))

def inspect_reid():
    datasets = {
        "Market-1501": {
            "path": MARKET_DIR,
            "subsets": ["bounding_box_train", "bounding_box_test", "query"]
        },
        "DukeMTMC": {
            "path": DUKE_DIR,
            "subsets": ["bounding_box_train", "bounding_box_test", "query"]
        }
    }
    
    stats = {}
    print("\n" + "="*40)
    print("   THỐNG KÊ DỮ LIỆU REID")
    print("="*40)
    print(f"{'Dataset':<15} | {'Subset':<20} | {'Số lượng ảnh'}")
    print("-" * 55)

    for name, info in datasets.items():
        stats[name] = []
        for sub in info["subsets"]:
            full_path = os.path.join(info["path"], sub)
            count = count_files(full_path)
            stats[name].append(count)
            
            # Debug nếu count = 0
            status_msg = str(count)
            if count == 0:
                status_msg += " ⚠️ (Check path!)"
                
            print(f"{name:<15} | {sub:<20} | {status_msg}")
            
            # In đường dẫn nếu lỗi để debug
            if count == 0 and sub == "bounding_box_train":
                print(f"   -> Đang tìm tại: {os.path.abspath(full_path)}")
    
    return stats

def inspect_mot():
    print("\n" + "="*40)
    print("   THỐNG KÊ DỮ LIỆU MOT17")
    print("="*40)
    
    if not os.path.exists(MOT_DIR):
        print(f"[CẢNH BÁO] Không tìm thấy thư mục MOT17 tại: {MOT_DIR}")
        return {}

    seqs = [d for d in os.listdir(MOT_DIR) if os.path.isdir(os.path.join(MOT_DIR, d))]
    seqs.sort()
    
    seq_stats = {}
    print(f"{'Sequence Name':<20} | {'Frames':<10} | {'GT File'}")
    print("-" * 50)
    
    for seq in seqs:
        seq_path = os.path.join(MOT_DIR, seq)
        img_path = os.path.join(seq_path, "img1")
        gt_path = os.path.join(seq_path, "gt", "gt.txt")
        
        n_frames = count_files(img_path, "*.jpg")
        seq_stats[seq] = n_frames
        
        has_gt = "✅ Có" if os.path.exists(gt_path) else "❌ Thiếu"
        print(f"{seq:<20} | {n_frames:<10} | {has_gt}")
        
    return seq_stats

def visualize_and_save(reid_stats, mot_stats):
    # Tạo folder result nếu chưa có
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    plt.figure(figsize=(16, 8))

    # 1. Biểu đồ ReID
    plt.subplot(1, 2, 1)
    labels = ["Train", "Test", "Query"]
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = plt.bar(x - width/2, reid_stats["Market-1501"], width, label='Market-1501', color='royalblue')
    rects2 = plt.bar(x + width/2, reid_stats["DukeMTMC"], width, label='DukeMTMC', color='darkorange')
    
    plt.ylabel('Số lượng ảnh')
    plt.title('Thống kê dữ liệu ReID')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Thêm số liệu
    for rect in rects1 + rects2:
        height = rect.get_height()
        if height > 0:
            plt.text(rect.get_x() + rect.get_width()/2., height,
                    f'{height}', ha='center', va='bottom', fontsize=9)

    # 2. Biểu đồ MOT17
    if mot_stats:
        plt.subplot(1, 2, 2)
        names = list(mot_stats.keys())
        counts = list(mot_stats.values())
        
        y_pos = np.arange(len(names))
        plt.barh(y_pos, counts, color='mediumseagreen')
        plt.yticks(y_pos, names, fontsize=8)
        plt.xlabel('Số lượng Frame')
        plt.title('Số lượng Frames trong MOT17')
        
        for i, v in enumerate(counts):
            plt.text(v + 10, i, str(v), va='center', fontsize=8)

    plt.tight_layout()
    
    # LƯU ẢNH
    save_path = os.path.join(RESULT_DIR, "data_statistics.png")
    plt.savefig(save_path)
    print(f"\n[SUCCESS] Đã lưu biểu đồ thống kê tại: {save_path}")
    
    # Hiển thị (nếu muốn tắt thì comment dòng dưới)
    # plt.show() 

if __name__ == "__main__":
    reid_data = inspect_reid()
    mot_data = inspect_mot()
    visualize_and_save(reid_data, mot_data)