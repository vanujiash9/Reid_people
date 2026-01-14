import os
import yaml
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['TORCH_HOME'] = "weights/torch_cache"

from src.data_loader.reid_loader import get_dataloaders
from src.models.reid_model import ReIDResNet50
from src.engine.trainer import ReIDTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Config
    cfg = load_config("config/reid_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Data Loaders (Nhớ chỉnh workers: 0 trong file config/reid_config.yaml trước khi chạy)
    train_loader, val_loader, num_classes = get_dataloaders(cfg)
    
    # 3. Khởi tạo Model
    model = ReIDResNet50(num_classes=num_classes, pretrained=cfg['model']['pretrained'])
    
    # --- LOGIC RESUME (TIẾP TỤC TRAIN) ---
    checkpoint_path = os.path.join(cfg['train']['output_dir'], "reid_best.pth")
    start_epoch = 1
    
    if os.path.exists(checkpoint_path):
        print(f"\n[RESUME] Tìm thấy checkpoint tại {checkpoint_path}")
        print(">>> Đang nạp trọng số và chuẩn bị chạy tiếp từ Epoch 16...")
        # Nạp trọng số đã lưu
        model.load_state_dict(torch.load(checkpoint_path))
        start_epoch = 16 
    
    model = model.to(device)
    
    # 4. Khởi tạo Trainer
    trainer = ReIDTrainer(cfg, model, train_loader, val_loader)
    
    # 5. ĐỒNG BỘ HÓA SCHEDULER
    # Phải cho scheduler chạy qua 15 bước để Learning Rate khớp với Epoch 16
    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            trainer.scheduler.step()
        print(f"[INFO] Scheduler updated. Current LR: {trainer.optimizer.param_groups[0]['lr']}")

    # 6. Chạy vòng lặp huấn luyện từ start_epoch
    run_resume(trainer, start_epoch)

def run_resume(trainer, start_epoch):
    print(f"\n>>> BẮT ĐẦU CHẠY TỪ EPOCH {start_epoch} TRÊN {trainer.device} (FP16 Enabled)")
    total_start_time = time.time()
    
    for epoch in range(start_epoch, trainer.cfg['train']['epochs'] + 1):
        epoch_start_time = time.time()
        
        # Train & Validate
        train_loss, train_acc = trainer.train_one_epoch(epoch)
        val_loss, val_acc = trainer.validate()
        
        # Cập nhật Learning Rate
        trainer.scheduler.step()
        
        duration = time.time() - epoch_start_time
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # In kết quả đầy đủ
        print(f"Epoch [{epoch}/{trainer.cfg['train']['epochs']}] "
              f"T_Loss: {train_loss:.4f} | T_Acc: {train_acc*100:.2f}% | "
              f"V_Loss: {val_loss:.4f} | V_Acc: {val_acc*100:.2f}% | "
              f"Time: {duration:.1f}s")

        # Lưu Log
        trainer.logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, current_lr, duration)
        
        # Lưu Checkpoint nếu tốt hơn
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.patience_counter = 0
            torch.save(trainer.model.state_dict(), os.path.join(trainer.output_dir, "reid_best.pth"))
            print("   >>> Saved Best Model (New Record!)")
        else:
            trainer.patience_counter += 1
            print(f"   >>> No improve. Patience: {trainer.patience_counter}/{trainer.cfg['train']['patience']}")
            
        if trainer.patience_counter >= trainer.cfg['train']['patience']:
            print(f"\n[STOP] Early Stopping: Model đã đạt giới hạn tại Epoch {epoch}.")
            break
            
    # Kết thúc
    trainer.logger.plot_charts()
    total_duration = (time.time() - total_start_time) / 60
    print(f"\n[DONE] Đã hoàn thành thêm {total_duration:.1f} phút huấn luyện.")

if __name__ == "__main__":
    main()