import torch
import torch.nn as nn
from torchvision.models import vision_transformer

def get_vit_tiny(num_classes=10, input_channels=3, image_size=32, patch_size=4):
    """
    Vision Transformer (ViT) adapted for CIFAR-10.
    Uses a small configuration to stay within research resource limits.
    """
    # ViT-Tiny configuration (roughly equivalent to DeiT-Tiny)
    # 12 layers, 192 hidden dim, 3 heads
    model = vision_transformer.VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=6,          # Reduced layers for speed
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=num_classes
    )
    
    # Adapt for input channels if not 3
    if input_channels != 3:
        model.conv_proj = nn.Conv2d(input_channels, 192, kernel_size=patch_size, stride=patch_size)
        
    return model

if __name__ == '__main__':
    # Test model instantiation and forward pass
    model = get_vit_tiny()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"ViT-Tiny Output Shape: {y.shape}")
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
