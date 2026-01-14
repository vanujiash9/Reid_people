import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm
# Cập nhật import cho PyTorch bản mới
from torch.amp import GradScaler, autocast 

from src.losses.cross_entropy import build_crossentropy_loss
from src.losses.triplet_loss import TripletLoss
from src.utils.logger import ExperimentLogger

class ReIDTrainer:
    def __init__(self, cfg, model, train_loader, val_loader):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Losses
        self.ce_loss = build_crossentropy_loss(model.classifier.out_features).to(self.device)
        self.triplet_loss = TripletLoss(margin=0.3).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'], 
                                    weight_decay=cfg['train']['weight_decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        
        # Logger
        self.logger = ExperimentLogger(cfg['train']['log_dir'])
        
        # Early Stopping vars
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.output_dir = cfg['train']['output_dir']
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        # Scaler
        self.scaler = GradScaler('cuda')

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg['train']['epochs']}", leave=False)
        for imgs, pids in pbar:
            imgs, pids = imgs.to(self.device), pids.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Cập nhật cú pháp mới: device_type='cuda'
            with autocast(device_type='cuda'):
                cls_score, global_feat = self.model(imgs)
                loss_id = self.ce_loss(cls_score, pids)
                loss_tri = self.triplet_loss(global_feat, pids)
                total_loss = loss_id + loss_tri

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += total_loss.item()
            
            _, preds = torch.max(cls_score, 1)
            correct += (preds == pids).sum().item()
            total += pids.size(0)
            
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, pids in self.val_loader:
                imgs, pids = imgs.to(self.device), pids.to(self.device)
                
                # Trick tính loss khi eval
                self.model.train() 
                with autocast(device_type='cuda'):
                    cls_score, global_feat = self.model(imgs)
                self.model.eval()
                
                # --- SỬA Ở ĐÂY: Thêm .float() ---
                loss = self.ce_loss(cls_score.float(), pids) + self.triplet_loss(global_feat.float(), pids)
                val_loss += loss.item()
                
                _, preds = torch.max(cls_score, 1)
                correct += (preds == pids).sum().item()
                total += pids.size(0)
                
        return val_loss / len(self.val_loader), correct / total

    def run(self):
        print(f">>> Start Training on {self.device} (FP16 Enabled)")
        start_time = time.time()
        
        for epoch in range(1, self.cfg['train']['epochs'] + 1):
            epoch_start = time.time()
            
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            
            duration = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch [{epoch}/{self.cfg['train']['epochs']}] "
                  f"T_Loss: {train_loss:.4f} | T_Acc: {train_acc*100:.2f}% | "
                  f"V_Loss: {val_loss:.4f} | V_Acc: {val_acc*100:.2f}% | "
                  f"Time: {duration:.1f}s")

            self.logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, current_lr, duration)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "reid_best.pth"))
                print("   >>> Saved Best Model (New Low Val_Loss)")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.cfg['train']['patience']:
                print(f"\n[STOP] Early Stopping: No improve for {self.cfg['train']['patience']} epochs.")
                break
                
        self.logger.plot_charts()
        total_time = (time.time() - start_time) / 60
        print(f"\n[DONE] Training Finished in {total_time:.1f} minutes.")
        print(f"Logs saved to: {self.cfg['train']['log_dir']}/training_log.csv")