from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from .pose_decoder import PoseDecoder
from .pose_cnn import PoseCNN
from .dem import DEM
from .spm import SPM

def get_model(num_layers=18, pretrained=True, **kwargs):
    """Create CADepth model"""
    encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
    decoder = DepthDecoder(encoder)
    return encoder, decoder

__all__ = ['ResnetEncoder', 'DepthDecoder', 'PoseDecoder', 'PoseCNN', 'DEM', 'SPM', 'get_model']
