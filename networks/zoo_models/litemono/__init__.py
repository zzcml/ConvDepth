from .resnet_encoder import ResnetEncoder
from .depth_encoder import DepthEncoder
from .depth_decoder import DepthDecoder
from .pose_decoder import PoseDecoder

def get_model(num_layers=18, pretrained=True, **kwargs):
    """Create Lite-Mono model"""
    encoder = DepthEncoder(num_layers=num_layers, pretrained=pretrained)
    decoder = DepthDecoder(encoder)
    return encoder, decoder

__all__ = ['ResnetEncoder', 'DepthEncoder', 'DepthDecoder', 'PoseDecoder', 'get_model']
