import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # --- FIX QUAN TRỌNG: Ép kiểu về float32 ---
        # Điều này giải quyết lỗi "Float and Half" và giúp tính khoảng cách chính xác hơn
        inputs = inputs.float() 
        
        n = inputs.size(0)
        # Tính khoảng cách Euclidean giữa các vector
        # dist = x^2 + y^2 - 2xy
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # Hard Mining:
        # Với mỗi ảnh (Anchor), tìm ảnh cùng ID xa nhất (Hard Positive)
        # và ảnh khác ID gần nhất (Hard Negative)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        dist_ap, _ = torch.max(dist * mask.float(), dim=1)
        dist_an, _ = torch.min(dist * (1 - mask.float()) + 1e6 * mask.float(), dim=1)
        
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss