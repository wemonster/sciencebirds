import os
import numpy as np

import torch

from PIL import Image
from tqdm import trange
import torchvision.transforms as transform
from .base import BaseDataset

class SciencebirdSeg(BaseDataset):
	NUM_CLASS = 14
	CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
		1, 64, 20, 63, 7, 72]
	def __init__(self, root='dataset', split='train',
				 mode=None, transform=None, target_transform=None,label_transform=None, **kwargs):
		super(SciencebirdSeg, self).__init__(
			root, split, mode, transform, target_transform, **kwargs)
		
		if self.mode == 'train':
			print ('train set')
			self.ids_file = os.path.join(root,'ImageSets/train.txt')
		elif self.mode == 'test':
			print ("test set")
			self.ids_file = os.path.join(root,'ImageSets/test.txt')
		else:
			print ('val set')
			self.ids_file = os.path.join(root,'ImageSets/val.txt')
		self.ids = self._load_image_set_index()
		self.image_files = os.path.join(root,'images')
		self.mask_files = os.path.join(root,'masks')
		self.label_files = os.path.join(root,'labels')
		# if split == 'train':
		# 	print('train set')
		# 	ann_file = os.path.join(root, 'annotations/instances_train2017.json')
		# 	ids_file = os.path.join(root, 'annotations/train_ids.pth')
		# 	self.root = os.path.join(root, 'train2017')
		# 	self.ids = open()
		# else:
		# 	print('val set')
		# 	ann_file = os.path.join(root, 'annotations/instances_val2017.json')
		# 	ids_file = os.path.join(root, 'annotations/val_ids.pth')
		# 	self.root = os.path.join(root, 'val2017')
		# self.coco = COCO(ann_file)
		# self.coco_mask = mask
		# self.mask = 
		# if os.path.exists(ids_file):
		# 	self.ids = torch.load(ids_file)
		# else:
		# 	ids = list(self.coco.imgs.keys())
		# 	self.ids = self._preprocess(ids, ids_file)
		self.transform = transform
		self.target_transform = target_transform
		self.label_transform = label_transform

	def _load_image_set_index(self):
		with open(self.ids_file,'r') as f:
			image_index = [int(x.strip()) for x in f.readlines()]
		return image_index

	def __getitem__(self, index):
		# coco = self.coco
		img_id = self.ids[index]
		img = Image.open(os.path.join(self.image_files, str(img_id)+'.png')).convert('RGB')
		# cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
		# mask = Image.fromarray(self._gen_seg_mask(
		# 	cocotarget, img_metadata['height'], img_metadata['width']))
		mask = Image.open(os.path.join(self.mask_files,str(img_id)+'.png')).convert('RGB')
		labels = Image.open(os.path.join(self.label_files,str(img_id)+'.png'))
		# synchrosized transform
		# if self.mode == 'train':
		# 	img, mask = self._sync_transform(img, mask)
		# elif self.mode == 'val':
		# 	img, mask = self._val_sync_transform(img, mask)
		# else:
		# 	assert self.mode == 'testval'
		# 	mask = self._mask_transform(mask)
		# general resize, normalize and toTensor
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			mask = self.target_transform(mask)
		# print (labels.size,type(labels))
		if self.label_transform is not None:
			labels = self.label_transform(labels)
		# print (labels.size(),type(labels))
		return img,labels

	def __len__(self):
		return len(self.ids)

	def _gen_seg_mask(self, target, h, w):
		mask = np.zeros((h, w), dtype=np.uint8)
		coco_mask = self.coco_mask
		for instance in target:
			rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
			m = coco_mask.decode(rle)
			cat = instance['category_id']
			if cat in self.CAT_LIST:
				c = self.CAT_LIST.index(cat)
			else:
				continue
			if len(m.shape) < 3:
				mask[:, :] += (mask == 0) * (m * c)
			else:
				mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
		return mask

	def _preprocess(self, ids, ids_file):
		print("Preprocessing mask, this will take a while." + \
			"But don't worry, it only run once for each split.")
		tbar = trange(len(ids))
		new_ids = []
		for i in tbar:
			img_id = ids[i]
			cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
			img_metadata = self.coco.loadImgs(img_id)[0]
			mask = self._gen_seg_mask(cocotarget, img_metadata['height'], 
									  img_metadata['width'])
			# more than 1k pixels
			if (mask > 0).sum() > 1000:
				new_ids.append(img_id)
			tbar.set_description('Doing: {}/{}, got {} qualified images'.\
				format(i, len(ids), len(new_ids)))
		print('Found number of qualified images: ', len(new_ids))
		torch.save(new_ids, ids_file)
		return new_ids
"""
NUM_CHANNEL = 91
[] background
[5] airplane
[2] bicycle
[16] bird
[9] boat
[44] bottle
[6] bus
[3] car
[17] cat
[62] chair
[21] cow
[67] dining table
[18] dog
[19] horse
[4] motorcycle
[1] person
[64] potted plant
[20] sheep
[63] couch
[7] train
[72] tv
"""
