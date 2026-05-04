from .resnet_encoder import ResnetEncoder
from .HR_Depth_Decoder import HRDepthDecoder
from .pose_decoder import PoseDecoder
from .pose_cnn import PoseCNN

def get_model(num_layers=18, pretrained=True, **kwargs):
    """Create DIFFNet model"""
    encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
    decoder = HRDepthDecoder(encoder)
    return encoder, decoder

__all__ = ['ResnetEncoder', 'HRDepthDecoder', 'PoseDecoder', 'PoseCNN', 'get_model']
