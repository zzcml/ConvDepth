from .resnet_encoder import ResnetEncoder
from .hrnet_encoder import HREncoder
from .depth_decoder import DepthDecoder
from .depth_decoder_msf import DepthDecoderMSF
from .pose_decoder import PoseDecoder
from .pose_cnn import PoseCNN

def get_model(num_layers=18, pretrained=True, **kwargs):
    """Create RA-Depth model"""
    encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
    decoder = DepthDecoderMSF(encoder)
    return encoder, decoder

__all__ = ['ResnetEncoder', 'HREncoder', 'DepthDecoder', 'DepthDecoderMSF', 'PoseDecoder', 'PoseCNN', 'get_model']
