###########################################################################
# Created by: Jianan Yang
# Email: u7083746@anu.edu.au
# Copyright (c) 2020
###########################################################################


import os
import cv2
import numpy as np
import time
def batch_pix_accuracy(output, target):
	"""Batch Pixel Accuracy
	Args:
		predict: input 4D tensor
		target: label 3D tensor
	"""
	# _, predict = torch.max(output, 1)
	predict = output
	predict = predict.astype('int64')
	target = target.astype('int64')
	correct_classified = np.nonzero(predict==target)

	pixel_labeled = np.sum(target > 0)
	#print (pixel_labeled)
	#print (np.unique(predict),np.unique(target))
	pixel_correct = np.sum((predict == target)*(target > 0))
	# print (pixel_labeled,pixel_correct)
	assert pixel_correct <= pixel_labeled, \
		"Correct area should be smaller than Labeled"
	return pixel_correct, pixel_labeled,correct_classified

def object_intersection_union(output,target):
	predict = output
	predict = predict.astype('int64')
	target = target.astype('int64')
	predict[predict > 0] = 1
	target[target > 0] = 1
	predict = predict * (target > 0).astype(predict.dtype)
	intersection = predict * (predict == target)
	# areas of intersection and union
	area_inter, _ = np.histogram(intersection, bins=2)
	area_pred, _ = np.histogram(predict, bins=2)
	area_lab, _ = np.histogram(target, bins=2)
	area_union = area_pred + area_lab - area_inter
	return area_inter, area_union

def batch_intersection_union(output, target, nclass):
	"""Batch Intersection of Union
	Args:
		predict: input 4D tensor
		target: label 3D tensor
		nclass: number of categories (int)
	"""
	# _, predict = torch.max(output, 1)
	predict = output
	mini = 1
	maxi = nclass
	nbins = nclass
	predict = predict.astype('int64') + 1
	target = target.astype('int64') + 1
	predict = predict * (target > 0).astype(predict.dtype)
	intersection = predict * (predict == target)
	# areas of intersection and union
	area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
	area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
	area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
	area_union = area_pred + area_lab - area_inter
	assert (area_inter <= area_union).all(), \
		"Intersection area should be smaller than Union area"
	return area_inter, area_union


def edge_accuracy(truthroot,resultroot):
	truth_folder = os.path.join(truthroot,'edge')
	result_folder = os.path.join(resultroot,'edge')
	imgs = os.listdir(result_folder)
	total_correct = 0
	total_label = 0
	for img in imgs:
		truth_im = cv2.imread(os.path.join(truth_folder,img))
		result_im = cv2.imread(os.path.join(result_folder,img))
		# print (img,truth_im)
		pixel_correct,pixel_labeled,_ = batch_pix_accuracy(result_im, truth_im)
		
		total_correct += pixel_correct
		total_label += pixel_labeled
	# total_inter += inter
	# total_union += union
	pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
	# IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
	# mIoU = IoU.mean()
	print (pixAcc)
	return pixAcc

def objectness_accuracy(truthroot,resultroot,ratio):
	# truth_folder = os.path.join(truthroot,'masks')
	result_folder = os.path.join(resultroot,'objectness')
	imgs = os.listdir(result_folder)
	total_correct = 0
	total_label = 0
	total_inter = 0
	total_union = 0
	for img in imgs:
		truth_im = cv2.imread(os.path.join(truthroot,img))
		objectness = np.array(truth_im)
		objectness[objectness!=[0,0,0]] = 255
		result_im = cv2.imread(os.path.join(result_folder,img))
		pixel_correct,pixel_labeled,_ = batch_pix_accuracy(result_im, objectness)
		area_inter,area_union = object_intersection_union(result_im,objectness)
		total_correct += pixel_correct
		total_label += pixel_labeled
		total_inter += area_inter
		total_union += area_union
	pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
	IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
	mIoU = IoU.mean()
	print (pixAcc,mIoU)
	return pixAcc,mIoU	

def object_box_accuracy(truthroot,resultroot):

	truth_folder = os.path.join(truthroot,'mask')
	result_folder = os.path.join(resultroot,'objectness')
	imgs = os.listdir(result_folder)
	for img in img_files:
		#boxes from groundtruth
		box_truth = img.split('.')[0]
		box_truth = os.path.join(truth_folder,box_truth+".txt")
		for t in open(box_truth,'r').readlines():
			info = t.split('|')
			X = int(info[0])
			Y = int(info[1])
			height = int(info[2])
			width = int(info[3])
			vertices = info[4]
			game_type = str(info[5]).strip().split('.')[1]


		#boxes from results
		masks = cv2.imread(os.path.join(result_folder,img),0)
		categories = np.unique(masks)
		colored = cv2.imread(os.path.join('dataset/rawdata/foregrounds',img))
		edge = cv2.Canny(colored,0,255)
		boxes_pred = []
		for cat in categories:
			if cat == 0:
				continue
			to_ret = np.zeros((480,840)).astype(np.uint8)
			to_ret[(masks==cat)] = 255
			to_ret[(edge==255)] = 0
			labels,num = skimage.measure.label(to_ret,connectivity=2,return_num=True)
			props = regionprops(labels)
			for prop in props:
				xmin,ymin,xmax,ymax = prop['bbox']
				if (xmax-xmin) * (ymax-ymin) < 30:
					continue
				boxes.append([xmin,ymin,xmax,ymax])

if __name__ == "__main__":
	groundtruth_folder = "dataset/rawdata/groundtruth"
	ratio_truth_folder = "dataset/images/small"
	test_image_folder = "../testresults"

	openness = [0,0.1,0.2,0.3,0.4,0.5]

	for ratio in openness:
		target_folder = int(ratio*10)
		truthroot = os.path.join(ratio_truth_folder,str(int(ratio*10)))
		resultroot = os.path.join(test_image_folder,str(int(ratio*10)))
		#edge
		edge_accuracy(truthroot,resultroot)
		#objectness
		objectness_accuracy("dataset/rawdata/foregrounds",resultroot,ratio)
		# #detection
		# object_box_accuracy(groundtruth_folder,resultroot)

