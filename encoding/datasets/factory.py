


"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from encoding.datasets.angrybird import angry_bird
import numpy as np


def init_imdb(category):
	for split in ['train','test','val']:
		name = 'angrybird_{}'.format(split)
		__sets[name] = (lambda split=split:angry_bird(split,category))

def get_imdb(name):
	"""Get an imdb (image database) by name."""
	name = "angrybird_{}".format(name)
	return __sets[name]()


def list_imdbs():
	"""List all registered imdbs."""
	return list(__sets.keys())
