import torch
import torch.nn as nn
from torchvision import models

class ReIDResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ReIDResNet50, self).__init__()
        print(f"[MODEL] Initializing ResNet50 for {num_classes} classes...")
        
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Trick: Last Stride = 1 (Giữ độ phân giải cao 16x8 thay vì 8x4)
        self.backbone.layer4[0].conv2.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)

        self.in_features = self.backbone.fc.in_features
        del self.backbone.fc, self.backbone.avgpool

        # BNNeck Structure
        self.bottleneck = nn.BatchNorm1d(self.in_features)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_features, num_classes, bias=False)

        self.bottleneck.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        global_feat = x.view(x.size(0), -1) 
        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat # Trả về cả 2 để tính (CrossEntropy + Triplet)
        else:
            return feat # Inference chỉ cần feature

    @staticmethod
    def weights_init_kaiming(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        elif classname.find('BatchNorm1d') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def weights_init_classifier(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)