import os
import psutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time

class ExperimentLogger:
    def __init__(self, log_dir):
        # Tạo folder logs nếu chưa có
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        self.logs = []
        print(f"[LOGGER] Log file will be saved to: {self.csv_path}")

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, lr, duration):
        # 1. Đo RAM hệ thống (%)
        ram_percent = psutil.virtual_memory().percent
        
        # 2. Đo VRAM GPU (GB)
        vram_gb = 0.0
        if torch.cuda.is_available():
            # Lấy lượng bộ nhớ PyTorch đang giữ (Reserved)
            vram_gb = torch.cuda.memory_reserved(0) / (1024**3)

        # 3. Tạo bản ghi
        record = {
            "Epoch": epoch,
            "Train_Loss": round(train_loss, 4),
            "Val_Loss": round(val_loss, 4),
            "Train_Acc": round(train_acc, 4),
            "Val_Acc": round(val_acc, 4),
            "Learning_Rate": lr,
            "Time_Sec": round(duration, 2),
            "RAM_Usage_Percent": ram_percent,
            "VRAM_Usage_GB": round(vram_gb, 2)
        }
        self.logs.append(record)
        
        # 4. Lưu ngay lập tức vào file CSV
        df = pd.DataFrame(self.logs)
        df.to_csv(self.csv_path, index=False)

    def plot_charts(self):
        if not self.logs: return
        df = pd.DataFrame(self.logs)
        epochs = df['Epoch']

        plt.figure(figsize=(12, 10))

        # Biểu đồ Loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, df['Train_Loss'], label='Train Loss', color='blue')
        plt.plot(epochs, df['Val_Loss'], label='Val Loss', color='red', linestyle='--')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Biểu đồ Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, df['Train_Acc'], label='Train Acc', color='green')
        plt.plot(epochs, df['Val_Acc'], label='Val Acc', color='orange', linestyle='--')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.log_dir, "training_charts.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[LOGGER] Charts saved to: {save_path}")