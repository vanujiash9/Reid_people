import torch
import numpy as np
from scipy.spatial.distance import cosine

class ReIDMatcher:
    def __init__(self, model, device, threshold=0.6):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.gallery_features = {} # {global_id: embedding}

    def get_embedding(self, img_crop):
        """Trích xuất vân tay từ ảnh người bị crop"""
        self.model.eval()
        with torch.no_grad():
            img_t = img_crop.to(self.device)
            feat = self.model(img_t)
            # Chuẩn hóa L2
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy()[0]

    def match(self, current_feat):
        """So sánh với kho dữ liệu để tìm ID cũ"""
        if not self.gallery_features:
            return None
        
        best_id = None
        min_dist = self.threshold
        
        for gid, feat in self.gallery_features.items():
            dist = cosine(current_feat, feat)
            if dist < min_dist:
                min_dist = dist
                best_id = gid
        return best_id

    def update_gallery(self, gid, feat):
        self.gallery_features[gid] = feat