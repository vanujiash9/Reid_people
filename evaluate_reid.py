import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.models.reid_model import ReIDResNet50
from src.engine.evaluator import ReIDEvaluator

# Cấu hình
DATA_DIR = "data/raw_data/Market-1501-v15.09.15" # Dùng bản Raw để test
MODEL_PATH = "weights/reid_best.pth"
BATCH_SIZE = 32

class MarketTestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.img_names = [f for f in os.listdir(folder_path) if f.endswith('.jpg') and not f.startswith('-1')]
        
    def __len__(self): return len(self.img_names)
    def __getitem__(self, idx):
        name = self.img_names[idx]
        # Format: pid_c1s1_...
        pid = int(name.split('_')[0])
        camid = int(name.split('_')[1][1])
        img = Image.open(os.path.join(self.folder_path, name)).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, pid, camid

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tfm = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load Data
    query_loader = DataLoader(MarketTestDataset(os.path.join(DATA_DIR, "query"), tfm), batch_size=BATCH_SIZE)
    gallery_loader = DataLoader(MarketTestDataset(os.path.join(DATA_DIR, "bounding_box_test"), tfm), batch_size=BATCH_SIZE)

    # 2. Load Model
    # Chú ý: num_classes lấy từ log của bạn là 1453
    model = ReIDResNet50(num_classes=1453)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)

    # 3. Tiến hành trích xuất và tính toán
    evaluator = ReIDEvaluator(model, device)
    
    print(">>> Đang trích xuất đặc trưng tập Query...")
    q_feat, q_pids, q_camids = evaluator.extract_features(query_loader)
    
    print(">>> Đang trích xuất đặc trưng tập Gallery...")
    g_feat, g_pids, g_camids = evaluator.extract_features(gallery_loader)

    print(">>> Đang tính toán Rank-1 và mAP...")
    rank1, mAP = evaluator.compute_metrics(q_feat, q_pids, q_camids, g_feat, g_pids, g_camids)

    print("\n" + "="*30)
    print(f"KẾT QUẢ CUỐI CÙNG TRÊN MARKET-1501:")
    print(f"  Rank-1 Accuracy: {rank1*100:.2f}%")
    print(f"  mAP Score:       {mAP*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate()