from .networks.resnet_encoder import ResnetEncoder
from .networks.depth_decoder import DepthDecoder
from .networks.pose_decoder import PoseDecoder
from .networks.pose_cnn import PoseCNN

def get_model(num_layers=18, pretrained=True, **kwargs):
    """Create TriDepth model"""
    encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
    decoder = DepthDecoder(encoder)
    return encoder, decoder

__all__ = ['ResnetEncoder', 'DepthDecoder', 'PoseDecoder', 'PoseCNN', 'get_model']
