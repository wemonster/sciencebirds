from .base import *

from .sciencebirds import SciencebirdSeg


datasets = {
    'sciencebirds':SciencebirdSeg
}

def get_segmentation_dataset(name,ratio,size='small', **kwargs):
    return datasets[name.lower()](ratio,size,**kwargs)
