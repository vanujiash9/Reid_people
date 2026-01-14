import torch.nn as nn

def build_crossentropy_loss(num_classes):
    return nn.CrossEntropyLoss(label_smoothing=0.1)