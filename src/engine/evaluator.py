import torch
import numpy as np
from tqdm import tqdm

class ReIDEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def extract_features(self, dataloader):
        self.model.eval()
        features = []
        pids = []
        camids = []
        
        with torch.no_grad():
            for imgs, labels, cams in tqdm(dataloader, desc="Extracting Features"):
                imgs = imgs.to(self.device)
                # Model trả về vector 2048 chiều
                feat = self.model(imgs) 
                features.append(feat.cpu())
                pids.extend(labels.numpy())
                camids.extend(cams.numpy())
                
        features = torch.cat(features, dim=0)
        # Chuẩn hóa L2 để tính Cosine Similarity chính xác hơn
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features.numpy(), np.array(pids), np.array(camids)

    def compute_metrics(self, q_feat, q_pids, q_camids, g_feat, g_pids, g_camids):
        """Tính Rank-1 và mAP theo chuẩn Market-1501"""
        # Tính khoảng cách Cosine (1 - cosine similarity)
        distmat = 1 - np.dot(q_feat, g_feat.T)
        
        num_q, num_g = distmat.shape
        indices = np.argsort(distmat, axis=1)
        
        all_cmc = []
        all_ap = []

        for i in range(num_q):
            # Lấy thứ tự các ảnh gallery gần query i nhất
            order = indices[i]
            
            # Loại bỏ ảnh có cùng PID và cùng CamID (theo chuẩn ReID)
            remove = (g_pids[order] == q_pids[i]) & (g_camids[order] == q_camids[i])
            keep = np.invert(remove)
            
            orig_cmc = (g_pids[order][keep] == q_pids[i])
            if not np.any(orig_cmc): continue

            # Tính CMC (Rank-1)
            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:10])

            # Tính Average Precision (AP)
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            prec = tmp_cmc / (np.arange(len(orig_cmc)) + 1)
            ap = (prec * orig_cmc).sum() / num_rel
            all_ap.append(ap)

        rank1 = np.array(all_cmc).mean(axis=0)[0]
        map_score = np.mean(all_ap)
        return rank1, map_score