import copy
import torch
import torch.nn as nn
import timm

from config import NUM_CLASSES, DROPOUT, MODEL_NAME, PRETRAINED


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=0)
    in_features = model.num_features
    model.classifier = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_gradcam_target_layer(model: nn.Module):
    # EfficientNet-B0: conv_pwl is the last pointwise projection conv in the
    # final MBConv block — best GradCAM signal before global average pooling.
    # Targeting the whole block gives coarser, weaker gradients.
    last_block = model.blocks[-1][-1]
    if hasattr(last_block, "conv_pwl"):
        return last_block.conv_pwl
    return model.blocks[-1][-1]  # fallback if arch differs


def copy_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def get_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
