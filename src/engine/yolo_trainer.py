import os
import time
import psutil
import torch
from ultralytics import YOLO
from src.utils.logger import ExperimentLogger


class YOLOTrainerEngine:
    def __init__(self, cfg):
        self.cfg = cfg

        # Load YOLO model
        self.model = YOLO(cfg['train']['model_type'])

        # Logger (dùng chung chuẩn với ReID)
        self.logger = ExperimentLogger(cfg['train']['log_dir'])
        self.logger.csv_path = os.path.join(
            cfg['train']['log_dir'],
            "yolo_training_log.csv"
        )

    def get_hw_stats(self):
        ram = psutil.virtual_memory().percent
        vram = 0.0
        if torch.cuda.is_available():
            vram = torch.cuda.memory_reserved(0) / (1024 ** 3)
        return ram, round(vram, 2)

    def train(self, data_yaml_path):
        ckpt_path = os.path.join(
            self.cfg['train']['output_dir'],
            "yolov8_mot17_results",
            "weights",
            "last.pt"
        )
        resume = os.path.exists(ckpt_path)

        print("\nSTARTING YOLO TRAINING")
        print(f"Model: {self.cfg['train']['model_type']}")
        print(f"Resume: {resume}")

        # ===================== CALLBACK =====================
        def on_train_epoch_end(trainer):
            epoch = trainer.epoch + 1

            # Metrics dictionary
            if hasattr(trainer.metrics, "results_dict"):
                metrics = trainer.metrics.results_dict
            else:
                metrics = trainer.metrics

            map50 = metrics.get("metrics/mAP50(B)", 0.0)
            map95 = metrics.get("metrics/mAP50-95(B)", 0.0)
            val_loss = metrics.get("val/box_loss", 0.0)

            # FIX CHUẨN: loss_items là Tensor
            train_loss = trainer.loss_items.mean().item()

            ram, vram = self.get_hw_stats()
            duration = (
                time.time() - trainer.epoch_start_time
                if hasattr(trainer, "epoch_start_time")
                else 0
            )

            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=map50,
                val_acc=map95,
                lr=trainer.optimizer.param_groups[0]["lr"],
                duration=duration,
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {train_loss:.4f} | "
                f"mAP50: {map50:.4f} | "
                f"mAP50-95: {map95:.4f} | "
                f"RAM: {ram:.1f}% | "
                f"VRAM: {vram:.2f}GB"
            )

        # Register callback
        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)

        # ===================== TRAIN =====================
        self.model.train(
            data=data_yaml_path,
            epochs=self.cfg["train"]["epochs"],
            imgsz=self.cfg["data"]["imgsz"],
            batch=self.cfg["train"]["batch_size"],
            patience=self.cfg["train"]["patience"],
            device=self.cfg["train"]["device"],
            workers=self.cfg["train"]["workers"],
            project=self.cfg["train"]["output_dir"],
            name="yolov8_mot17_results",
            exist_ok=True,
            cache=False,
            optimizer="AdamW",
            lr0=self.cfg["train"]["learning_rate"],
            resume=resume,
            plots=True,
        )

        self.logger.plot_charts()
        print("\nTRAINING FINISHED")
        print("Best model saved at:")
        print("weights/yolov8_mot17_results/weights/best.pt")
