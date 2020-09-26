from .base import *
from .fcn import *
from .deeplabv3 import *


def get_segmentation_model(ratio,nclass,name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'deeplab': get_deeplab
    }
    return models[name.lower()](ratio,nclass,**kwargs)
