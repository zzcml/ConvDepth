"""PackNet-SfM models"""
from .networks.depth import DepthNet
from .networks.pose import PoseNet

def get_model(version='1.0', **kwargs):
    """Create PackNet-SfM model"""
    depth_net = DepthNet(version=version, **kwargs)
    pose_net = PoseNet()
    return depth_net, pose_net

__all__ = ['DepthNet', 'PoseNet', 'get_model']
