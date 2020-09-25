# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
# from scipy.misc import imread
import cv2
from cv2 import imread
import pdb
from PIL import Image
from IPython.display import display

def get_minibatch(roidb, num_classes):
	"""Given a roidb, construct a minibatch sampled from it."""
	num_images = len(roidb)
	# Sample random scales to use for each image in this batch

	# Get the input image blob, formatted for caffe
	# im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

	blobs = {}

	# gt boxes: (x1, y1, x2, y2, cls)
	gt_inds = len(roidb[0]['boxes'])
	gt_boxes = np.empty((gt_inds, 5), dtype=np.int32)
	gt_boxes[:, 0] = roidb[0]['boxes'][:, 0]
	gt_boxes[:, 2] = roidb[0]['boxes'][:, 2]
	gt_boxes[:, 1] = roidb[0]['boxes'][:, 1]
	gt_boxes[:, 3] = roidb[0]['boxes'][:, 3]
	gt_boxes[:, 4] = roidb[0]['gt_classes'][:]

	blobs['gt_boxes'] = gt_boxes
	blobs['im_info'] = np.array([[480,840]],
								dtype=np.int32)

	blobs['img_id'] = roidb[0]['img_id']

	return blobs


def _get_image_blob(roidb, scale_inds):
	"""Builds an input blob from the images in the roidb at the specified
	scales.
	"""
	# num_images = len(roidb)
	num_images = 1

	processed_ims = []
	im_scales = []
	for i in range(num_images):
		im = imread(roidb[i]['image']) / 255
		
		if len(im.shape) == 2:
			im = im[:, :, np.newaxis]
			im = np.concatenate((im, im, im), axis=2)
		# flip the channel, since the original one using cv2
		# rgb -> bgr
		im = im[:, :, ::-1]

		if roidb[i]['flipped']:
			im = im[:, ::-1, :]
		target_size = cfg.TRAIN.SCALES
		# im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
		# 								cfg.TRAIN.MAX_SIZE)
		cv2.imwrite('1.jpg',im)
		im_scale = (1.0,1.0)
		im_scales.append(im_scale)
		processed_ims.append(im)

	# Create a blob to hold the input images
	#blob: num_images x H x W x D
	blob = im_list_to_blob(processed_ims)

	return blob, im_scales
