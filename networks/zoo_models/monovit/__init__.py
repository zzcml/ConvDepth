from .mpvit import mpvit_small
from .hr_decoder import DepthDecoder
from .nets import DeepNet

def get_model(model_type='mpvitnet', **kwargs):
    """Create MonoViT model"""
    model = DeepNet(type=model_type, **kwargs)
    return model

__all__ = ['mpvit_small', 'DepthDecoder', 'DeepNet', 'get_model']
