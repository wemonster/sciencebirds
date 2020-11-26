###########################################################################
# Created by: Jianan Yang
# Email: u7083746@anu.edu.au
# Copyright (c) 2020
###########################################################################

import random
import numpy as np

import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['BaseDataset', 'test_batchify_fn']

class BaseDataset(data.Dataset):
	def __init__(self, root, split, mode=None, transform=None, 
				 target_transform=None,label_transform=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.label_transform = label_transform
		self.split = split
		self.mode = mode if mode is not None else split

	def __getitem__(self, index):
		raise NotImplemented

	@property
	def num_class(self):
		return self.NUM_CLASS

	@property
	def pred_offset(self):
		raise NotImplemented

	def make_pred(self, x):
		return x + self.pred_offset

	def _mask_transform(self, mask):
		return torch.from_numpy(np.array(mask)).long()


def test_batchify_fn(data):
	error_msg = "batch must contain tensors, tuples or lists; found {}"
	if isinstance(data[0], (str, torch.Tensor)):
		return list(data)
	elif isinstance(data[0], (tuple, list)):
		data = zip(*data)
		return [test_batchify_fn(i) for i in data]
	raise TypeError((error_msg.format(type(data[0]))))
