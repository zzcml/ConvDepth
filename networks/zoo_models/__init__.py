"""
Zoo Models - Collection of depth estimation models from RoboDepth
"""

# Model registry
AVAILABLE_MODELS = {
    'cadepth': 'CADepth',
    'diffnet': 'DIFFNet',
    'dnet': 'DNet',
    'depthhints': 'DepthHints',
    'dynadepth': 'DynaDepth',
    'epcdepth': 'EPCDepth',
    'fsredepth': 'FSRE-Depth',
    'gcndepth': 'GCNDepth',
    'hrdepth': 'HR-Depth',
    'instadm': 'Insta-DM',
    'litemono': 'Lite-Mono',
    'manydepth': 'ManyDepth',
    'maskocc': 'MaskOcc',
    'monodepth2': 'MonoDepth2',
    'monovit': 'MonoViT',
    'packnetsfm': 'PackNet-SfM',
    'radepth': 'RA-Depth',
    'sgdepth': 'SGDepth',
    'tridepth': 'TriDepth',
}

def get_model(model_name, **kwargs):
    """
    Get a model from the zoo by name.
    
    Args:
        model_name: Name of the model (case-insensitive)
        **kwargs: Additional arguments for model construction
    
    Returns:
        Model instance or None if not found
    """
    model_name_lower = model_name.lower().replace('-', '').replace('_', '')
    
    if model_name_lower not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    try:
        module = __import__(f'networks.zoo_models.{model_name_lower}', fromlist=[''])
        if hasattr(module, 'get_model'):
            return module.get_model(**kwargs)
        else:
            # Try to find common model classes
            for attr_name in dir(module):
                if attr_name.lower() in ['model', 'network', 'depthnetwork', 'monodepth']:
                    return getattr(module, attr_name)(**kwargs)
            raise AttributeError(f"No get_model function or standard model class found in {model_name_lower}")
    except ImportError as e:
        raise ImportError(f"Failed to import model '{model_name}': {e}")

__all__ = ['AVAILABLE_MODELS', 'get_model']
