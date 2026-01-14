import os
import yaml
import torch

# Ép cache model vào ổ D
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), "weights", "torch_cache")

from src.data_loader.yolo_loader import YOLODataLoader
from src.engine.yolo_trainer import YOLOTrainerEngine

def main():
    # 1. Kiểm tra môi trường
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n>>> Running YOLO Train on: {device}")
    if torch.cuda.is_available():
        print(f">>> GPU: {torch.cuda.get_device_name(0)}")

    # 2. Đọc Config với UTF-8
    config_path = "config/yolo_config.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 3. Chuẩn bị Dữ liệu (Tạo file .yaml cho YOLO)
    loader = YOLODataLoader(cfg)
    data_yaml = loader.prepare_data_yaml()

    # 4. Chạy Trainer
    trainer = YOLOTrainerEngine(cfg)
    trainer.train(data_yaml)

if __name__ == "__main__":
    main()