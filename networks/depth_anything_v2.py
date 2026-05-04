"""
Depth-Anything-V2 Model Wrapper
Supports ViT-Small (vits) and ViT-Large (vitl) variants
Can be used as both teacher and student model in distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_anything_v2_dpt import DepthAnythingV2 as DAV2Model


class DepthAnythingV2(nn.Module):
    """
    Depth-Anything-V2 wrapper for training and inference.
    
    Args:
        variant (str): Model variant, either 'vits' (Small) or 'vitl' (Large)
        pretrained_path (str, optional): Path to pretrained weights
        use_teacher (bool): If True, freeze parameters for teacher model
    
    Input:
        x: Tensor of shape [N, 3, H, W], RGB images normalized to [0, 1]
        
    Output:
        depth: Tensor of shape [N, H, W], predicted depth map
    """
    
    def __init__(self, variant='vits', pretrained_path=None, use_teacher=False):
        super().__init__()
        
        assert variant in ['vits', 'vitl', 'vitb', 'vitg'], \
            f"Unknown variant: {variant}. Choose from vits, vitl, vitb, vitg"
        
        self.variant = variant
        
        # Create the base model
        self.model = DAV2Model(encoder=variant)
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
        
        # Freeze parameters if using as teacher model
        if use_teacher:
            self.freeze()
    
    def load_pretrained(self, path):
        """Load pretrained weights from checkpoint file."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove prefix if present (e.g., "module.")
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # Load weights with strict=False to handle minor mismatches
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights from {path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
    
    def freeze(self):
        """Freeze all parameters for teacher model."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [N, 3, H, W]
            
        Returns:
            depth: Depth map [N, H, W]
        """
        # Ensure input is in expected range [0, 1] or normalize if needed
        # DAV2 expects normalized RGB input
        
        # Get depth prediction
        depth = self.model(x)
        
        return depth
    
    @torch.no_grad()
    def infer(self, x):
        """Inference mode with additional post-processing."""
        self.eval()
        depth = self.forward(x)
        return depth


# Factory function for easy model creation
def create_depth_anything_v2(variant='vits', pretrained_path=None, is_teacher=False):
    """
    Create Depth-Anything-V2 model.
    
    Args:
        variant: 'vits' or 'vitl'
        pretrained_path: Path to weights
        is_teacher: Whether to freeze parameters
    
    Returns:
        model: DepthAnythingV2 instance
    """
    return DepthAnythingV2(
        variant=variant,
        pretrained_path=pretrained_path,
        use_teacher=is_teacher
    )
