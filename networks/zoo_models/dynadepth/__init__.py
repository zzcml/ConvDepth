from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from .gravity_decoder import GravityDecoder
from .velo_decoder import VeloDecoder
from .pose_decoder import PoseDecoder
from .pose_cnn import PoseCNN

def get_model(num_layers=18, pretrained=True, **kwargs):
    """Create DynaDepth model"""
    encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
    decoder = DepthDecoder(encoder)
    return encoder, decoder

__all__ = ['ResnetEncoder', 'DepthDecoder', 'GravityDecoder', 'VeloDecoder', 'PoseDecoder', 'PoseCNN', 'get_model']
