"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import encoding.datasets as datasets
import numpy as np
from encoding.datasets.factory import init_imdb,get_imdb
import PIL
import pdb


def prepare_roidb(imdb):
	"""Enrich the imdb's roidb by adding some derived quantities that
	are useful for training. This function precomputes the maximum
	overlap, taken over ground-truth boxes, between each ROI and
	each ground-truth box. The class with maximum overlap is also
	recorded.
	"""

	roidb = imdb.roidb
	for i in range(len(imdb.image_index)):
		roidb[i]['img_id'] = imdb.image_id_at(i)
		# need gt_overlaps as a dense array for argmax
		gt_overlaps = roidb[i]['overlaps'].toarray()
		# max overlap with gt over classes (columns)
		max_overlaps = gt_overlaps.max(axis=1)
		# gt class that had the max overlap
		max_classes = gt_overlaps.argmax(axis=1)
		roidb[i]['max_classes'] = max_classes
		roidb[i]['max_overlaps'] = max_overlaps
		# sanity checks
		# max overlap of 0 => class should be zero (background)
		zero_inds = np.where(max_overlaps == 0)[0]
		assert all(max_classes[zero_inds] == 0.0)



def filter_roidb(roidb):
	# filter the image without bounding box.
	print('before filtering, there are %d images...' % (len(roidb)))
	i = 0
	while i < len(roidb):
		if len(roidb[i]['boxes']) == 0:
			del roidb[i]
			i -= 1
		i += 1

	print('after filtering, there are %d images...' % (len(roidb)))
	return roidb


def combined_roidb(imdb_names, classes,training=True):
	"""
	Combine multiple roidbs
	"""
	def get_training_roidb(imdb):

		print('Preparing training data...')

		prepare_roidb(imdb)
		#ratio_index = rank_roidb_ratio(imdb)
		print('done')

		return imdb.roidb

	def get_roidb(imdb_name):
		imdb = get_imdb(imdb_name)
		print('Loaded dataset `{:s}` for training'.format(imdb.name))
		roidb = get_training_roidb(imdb)
		return roidb

	init_imdb(classes)
	roidbs = [get_roidb(s) for s in imdb_names.split('+')]
	print(imdb_names.split('+'))
	roidb = roidbs[0]

	if len(roidbs) > 1:
		for r in roidbs[1:]:
			roidb.extend(r)
		tmp = get_imdb(imdb_names.split('+')[1])
		imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
	else:
		imdb = get_imdb(imdb_names)

	if training:
		roidb = filter_roidb(roidb)
	return imdb, roidb