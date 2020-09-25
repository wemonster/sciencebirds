from __future__ import print_function
from __future__ import absolute_import


import xml.dom.minidom as minidom

import os
import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR


class angry_bird(imdb):
	def __init__(self,image_set,category):
		super(angry_bird,self).__init__(image_set)
		self._image_set = image_set
		self._data_path = 'dataset'
		self._category = category
		self._classes = category.gameObjectType.keys()
		self._image_ext = '.png'
		self._image_index = self._load_image_set_index()
		self._roidb = self.get_roidb()
		#self._roidb_handler = self.gt_roidb

	def image_path_at(self,i):
		return self.image_path_from_index(self._image_index[i])

	def image_id_at(self,i):
		return i

	def image_path_from_index(self,index):
		image_path = os.path.join(self._data_path,'images',str(index)+self._image_ext)
		return image_path

	def name_from_index(self,index):
		name = '0'*(6-len(str(index))) + str(index)
		return name

	def _load_image_set_index(self):
		image_set_file = os.path.join(self._data_path,'ImageSets',self._image_set+'.txt')
		with open(image_set_file) as f:
			image_index = [int(x.strip()) for x in f.readlines()]
		return image_index

	def get_roidb(self):
		gt_roidb = [self._load_annotations(index) for index in self.image_index]
		return gt_roidb

	def rpn_roidb(self):
		gt_roidb = self.get_roidb()
		rpn_roidb = self._load_rpn_roidb()
		roidb = imdb.merge_roidbs(gt_roidb,rpn_roidb)
		return roidb

	def _load_rpn_roidb(self):
		gt_roidb = self.get_roidb()
		boxlist = None
		return self.create_roidb_from_box_list(boxlist,gt_roidb)


	def _load_annotations(self,index):
		filename = os.path.join(self._data_path,'annotations',str(index)+'.xml')
		
		tree = ET.parse(filename)
		objs = tree.findall('object')
		num_objs = len(objs)

		boxes = np.zeros((num_objs,4),dtype=np.uint16) #box coordinates
		gt_classes = np.zeros((num_objs),dtype=np.int32) #which category it belongs to
		overlaps = np.zeros((num_objs,self.num_classes),dtype=np.float32) #IoU

		for ix, obj in enumerate(objs):
			bbox = obj.find('box')
			cls_cat = obj.find('name').text.strip()
			if cls_cat == 'UNKNOWN' and (self._image_set == 'train' or self._image_set == 'val'):
				continue
			x1 = float(bbox.find('xmin').text)
			y1 = float(bbox.find('ymin').text)
			x2 = float(bbox.find('xmax').text)
			y2 = float(bbox.find('ymax').text)
			
			cls_score = self._category.convert_class_to_category(cls_cat)
			boxes[ix,:] = [x1,y1,x2,y2]
			gt_classes[ix] = cls_score
			overlaps[ix,cls_score] = 1.0

		overlaps = scipy.sparse.csr_matrix(overlaps)
		return {'boxes':boxes,
				'gt_classes':gt_classes,
				'overlaps':overlaps,
				'flipped':False}

	def default_roidb(self):
		pass


	def evaluate_detections(self,all_boxes, output_dir=None):
		pass
