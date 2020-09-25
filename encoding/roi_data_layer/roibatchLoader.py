
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from encoding.utils.config import cfg
from encoding.roi_data_layer.minibatch import get_minibatch, get_minibatch

import numpy as np
import random
import time
import pdb


class roibatchLoader(data.Dataset):

	def __init__(self, roidb, imdb,num_classes,mode='train'):
		self._roidb = roidb
		self._num_classes = num_classes
		# we make the height of image consistent to trim_height, trim_width

		self.max_num_box = cfg.MAX_NUM_GT_BOXES

		self.mode = mode

		self._imdb = imdb

	def __getitem__(self, index):

		# get the anchor index for current sample index
		# here we set the anchor index to the last one
		# sample in this group
		minibatch_db = [self._roidb[index]]
		blobs = get_minibatch(minibatch_db, self._num_classes)
		im_info = torch.from_numpy(blobs['im_info'])
		im_id = blobs['img_id']
		img,labels,objectness = self._imdb[index]
		# we need to random shuffle the bouding box.
		if self.mode == 'train':
			np.random.shuffle(blobs['gt_boxes'])
			gt_boxes = torch.from_numpy(blobs['gt_boxes'])
			# print (gt_boxes.size)
			########################################################
			# padding the input image to fixed size for each group #
			########################################################

			# NOTE1: need to cope with the case where a group cover both conditions. (done)
			# NOTE2: need to consider the situation for the tail samples. (no worry)
			# NOTE3: need to implement a parallel data loader. (no worry)
			# get the index range

			# gt_boxes.clamp_(0, trim_size)
			# gt_boxes[:, :4].clamp_(0, trim_size)
			# check the bounding box:
			not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (
				gt_boxes[:, 1] == gt_boxes[:, 3])
			keep = torch.nonzero(not_keep == 0).view(-1)

			if keep.numel() != 0:
				gt_boxes = gt_boxes[keep]
				num_boxes = min(gt_boxes.size(0), self.max_num_box)
				gt_boxes = gt_boxes[:num_boxes,:]
			else:
				num_boxes = 0

			im_info = im_info.view(2)
			return im_info, gt_boxes, num_boxes,im_id
		else:

			im_info = im_info.view(2)

			gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
			num_boxes = 0

			return im_info, gt_boxes, num_boxes,im_id

	def __len__(self):
		return len(self._roidb)
