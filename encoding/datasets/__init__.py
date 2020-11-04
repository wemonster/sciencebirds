from .base import *

from .sciencebirds import SciencebirdSeg


datasets = {
    'sciencebirds':SciencebirdSeg
}

def get_segmentation_dataset(name,filename,size='small', **kwargs):
    return datasets[name.lower()](filename,size,**kwargs)
