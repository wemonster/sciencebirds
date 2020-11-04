import os
import numpy as np
import cv2
import copy
import torch

from PIL import Image
from tqdm import trange
import torchvision.transforms as transform
from .base import BaseDataset

class SciencebirdSeg(BaseDataset):
	def __init__(self, ratio,size='small', root='dataset/images',split='train',
				 mode=None, transform=None, target_transform=None,label_transform=None, **kwargs):
		super(SciencebirdSeg, self).__init__(
			root, split, mode, transform, target_transform, **kwargs)
		
		folder = os.path.join(root,"{}/{}".format(size,int(ratio*10)))
		if self.mode == 'train':
			print ('train set')
			self.ids_file = os.path.join("dataset",'ImageSets/train.txt')
		elif self.mode == 'test':
			print ("test set")
			self.ids_file = os.path.join("dataset",'ImageSets/test.txt')
			self.unknowns_files = "dataset/rawdata/foregroundmask"
		else:
			print ('val set')
			self.ids_file = os.path.join("dataset",'ImageSets/val.txt')
		self.ids = self._load_image_set_index()
		# self.image_files = "dataset/rawdata/groundtruthimage"
		self.image_files = os.path.join(folder,"foregrounds")
		self.label_files = os.path.join(folder,'masks')
		self.foreground_files = "dataset/rawdata/foregrounds"
		self.edge_files = os.path.join(folder,"edge")
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
		img_id = self.ids[index]
		img = Image.open(os.path.join(self.image_files, str(img_id)+'.png')).convert('RGB') #foregrounds only
		#img = np.array(img)
		#img[np.where((img!=[0,0,0]).all(axis=2))] = [255,255,255]
		labels = Image.open(os.path.join(self.label_files,str(img_id)+'.png'))
		edge = Image.open(os.path.join(self.edge_files,str(img_id) + '.png'))
		foregrounds = Image.open(os.path.join(self.foreground_files,str(img_id)+'.png')).convert('RGB')
		if self.mode == 'test':
			#foregrounds = np.array(foregrounds)
			#foregrounds[np.where((foregrounds!=[0,0,0]).all(axis=2))] = [255,255,255]
			labels = Image.open(os.path.join(self.unknowns_files,str(img_id)+'.png'))
		objectness = np.array(labels)
		objectness[objectness > 0] = 1
		# print (objectness.shape,labels.size,edge.size)
		edge = np.array(edge)
		edge[edge > 0] = 1
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			foregrounds = self.target_transform(foregrounds)
		if self.label_transform is not None:
			labels = self.label_transform(labels) * 255
			labels = labels.type(torch.LongTensor)
			objectness = self.label_transform(objectness) * 255
			objectness = objectness.type(torch.LongTensor)

			edge = self.label_transform(edge) * 255
			edge = edge.type(torch.LongTensor)
		if self.mode == 'test':
			return foregrounds,labels,objectness,edge
		return img,labels,objectness,edge

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
