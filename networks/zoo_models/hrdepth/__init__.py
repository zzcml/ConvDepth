from .resnet_encoder import ResnetEncoder
from .mobilenetV3_encoder import MobileNetV3Encoder
from .depth_decoder import DepthDecoder
from .HR_Depth_Decoder import HRDepthDecoder

def get_model(num_layers=18, pretrained=True, **kwargs):
    """Create HR-Depth model"""
    encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
    decoder = HRDepthDecoder(encoder)
    return encoder, decoder

__all__ = ['ResnetEncoder', 'MobileNetV3Encoder', 'DepthDecoder', 'HRDepthDecoder', 'get_model']
