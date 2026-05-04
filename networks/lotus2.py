"""
Lotus-2 Model Wrapper for Distillation
Based on the official Lotus-2 implementation using Diffusers Flux pipeline

Note: This model requires diffusers library and significant GPU memory.
Recommended for teacher model only due to its large size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Lotus2(nn.Module):
    """
    Lotus-2 wrapper for depth estimation as teacher model.
    
    This is a simplified wrapper that loads the full Lotus-2 pipeline
    and extracts depth predictions. Due to the complexity of the original
    implementation (based on Flux diffusion model), this wrapper handles
    the core functionality needed for distillation.
    
    Args:
        pretrained_path (str, optional): Path to pretrained Lotus-2 weights
        use_teacher (bool): Always True for this model (frozen teacher)
        variant (str): Task variant, 'depth' or 'normal'
    
    Input:
        x: Tensor of shape [N, 3, H, W], RGB images normalized to [0, 1]
        
    Output:
        depth: Tensor of shape [N, H, W], predicted depth map
    """
    
    def __init__(self, pretrained_path=None, use_teacher=True, variant='depth'):
        super().__init__()
        
        self.variant = variant
        self.pipeline = None
        self._model_loaded = False
        
        # Store config for lazy loading
        self.pretrained_path = pretrained_path
        
        # Always freeze as teacher model
        if use_teacher:
            self.freeze()
    
    def _load_pipeline(self):
        """Lazy load the diffusers pipeline when first needed."""
        if self._model_loaded:
            return
        
        try:
            from diffusers import FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
            from diffusers.pipelines.flux import FluxPipeline
            print("Loading Lotus-2 pipeline...")
            
            # Import local pipeline if available
            import sys
            import os
            lotus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Lotus-2')
            if os.path.exists(lotus_path):
                sys.path.insert(0, lotus_path)
                from pipeline import Lotus2Pipeline
            else:
                # Fall back to base FluxPipeline
                Lotus2Pipeline = FluxPipeline
            
            # Load model
            if self.pretrained_path:
                self.pipeline = Lotus2Pipeline.from_pretrained(
                    self.pretrained_path,
                    torch_dtype=torch.float16
                )
            else:
                # Default HuggingFace repo
                self.pipeline = Lotus2Pipeline.from_pretrained(
                    "jingheya/Lotus-2",
                    torch_dtype=torch.float16
                )
            
            self.pipeline = self.pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')
            self._model_loaded = True
            print("Lotus-2 pipeline loaded successfully")
            
        except ImportError as e:
            print(f"Error: Failed to import diffusers or Lotus-2 dependencies: {e}")
            print("Please install: pip install diffusers transformers accelerate")
            raise
        except Exception as e:
            print(f"Error loading Lotus-2: {e}")
            raise
    
    def freeze(self):
        """Freeze all parameters for teacher model."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self, x):
        """
        Forward pass through Lotus-2.
        
        Args:
            x: Input tensor [N, 3, H, W]
            
        Returns:
            depth: Depth map [N, H, W]
        """
        if not self._model_loaded:
            self._load_pipeline()
        
        self.eval()
        with torch.no_grad():
            # Resize input to expected size if needed
            input_h, input_w = x.shape[2:]
            target_size = 518  # Standard Lotus-2 input size
            
            if input_h != target_size or input_w != target_size:
                x_resized = F.interpolate(x, size=(target_size, target_size), 
                                         mode='bilinear', align_corners=False)
            else:
                x_resized = x
            
            # Run inference through pipeline
            # Note: Original Lotus-2 uses a complex diffusion process
            # Here we simplify for distillation purposes
            
            if hasattr(self.pipeline, '__call__'):
                # Use the custom Lotus2Pipeline if available
                output = self.pipeline(
                    rgb_in=x_resized,
                    prompt="",
                    num_inference_steps=10,
                    output_type="np",
                    process_res=target_size,
                    timestep_core_predictor=1,
                    guidance_scale=3.5,
                    return_dict=False
                )
                depth_pred = output[0]
                
                # Convert to tensor if numpy
                if isinstance(depth_pred, list):
                    depth_pred = torch.stack([torch.from_numpy(d) for d in depth_pred])
                elif not isinstance(depth_pred, torch.Tensor):
                    depth_pred = torch.from_numpy(depth_pred)
                
                # Handle channel dimension
                if depth_pred.dim() == 4 and depth_pred.shape[1] == 1:
                    depth_pred = depth_pred.squeeze(1)
                elif depth_pred.dim() == 4 and depth_pred.shape[-1] == 1:
                    depth_pred = depth_pred.permute(0, 3, 1, 2).squeeze(1)
                    
            else:
                # Fallback: simple placeholder (should not happen in practice)
                print("Warning: Using fallback depth estimation")
                depth_pred = torch.zeros_like(x[:, 0])
            
            # Resize back to original input size
            if input_h != target_size or input_w != target_size:
                depth_pred = F.interpolate(
                    depth_pred.unsqueeze(1) if depth_pred.dim() == 3 else depth_pred,
                    size=(input_h, input_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            return depth_pred
    
    @torch.no_grad()
    def infer(self, x):
        """Inference mode."""
        return self.forward(x)


# Factory function
def create_lotus2(pretrained_path=None, is_teacher=True):
    """
    Create Lotus-2 model.
    
    Args:
        pretrained_path: Path to weights or HuggingFace repo ID
        is_teacher: Whether to freeze parameters (always True for Lotus-2)
    
    Returns:
        model: Lotus2 instance
    """
    return Lotus2(
        pretrained_path=pretrained_path,
        use_teacher=is_teacher
    )
