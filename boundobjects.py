###########################################################################
# Created by: Jianan Yang
# Email: u7083746@anu.edu.au
# Copyright (c) 2020
###########################################################################

import numpy as np

import skimage.measure
from skimage.measure import regionprops
import cv2
import os
from classlabel import Category
# masks = cv2.imread("dataset/images/small/0/masks/0.png",0)

# img = cv2.imread("dataset/rawdata/foregrounds/1.png")
# edge = cv2.Canny(img,0,255)
# dilated_edge = cv2.dilate(edge,np.ones((2,2)))
# cv2.imwrite("edges.png",dilated_edge)
# masks = cv2.imread("../testresults/0/1.png",0)
# masks[dilated_edge==255] = 0
# cv2.imwrite("cutted.png",masks)
def get_class_lists():
	data = open("logs/resnet.txt",'r').readlines()
	class_info = []
	for i in data:
		filename,classes = i.split('|')
		classes = classes.strip().split(',')
		class_info.append((filename,classes))
	return class_info
outdir = "../sciencebirdoutputs/ICE"
if not os.path.exists(outdir):
	os.makedirs(outdir)
pallete_folder = "../sciencebirdoutputs/ICE/pallete"
if not os.path.exists(pallete_folder):
	os.makedirs(pallete_folder)
# image_folder = "../testresults/ICE/mask"
image_folder = "../experiments/results/ICE/mask"
img_files = os.listdir(image_folder)
class_info = get_class_lists()

category = Category(class_info[17][1],True)
print (category.id_to_cat.keys())
colormap = {
		'BACKGROUND':[0,0,0],
		'BLACKBIRD':[128,0,0],
		'BLUEBIRD':[0,128,0],
		'HILL':[128,128,0],
		'ICE':[0,0,128],
		'PIG':[128,0,128],
		'REDBIRD':[0,128,128],
		'STONE':[128,128,128],
		'WHITEBIRD':[64,0,0],
		'WOOD':[192,0,0],
		'YELLOWBIRD':[64,128,128],
		'SLING':[192,128,128],
		'TNT':[64,128,128],
		'UNKNOWN':[255,255,255]
	}

for img in img_files:
	if img.endswith('png'):
		masks = cv2.imread(os.path.join(image_folder,img),0)
		print (masks)
		categories = np.unique(masks)
		print (categories)
		pallete = np.zeros((480,840,3)).astype(np.uint8)
		id_to_cat = category.id_to_cat
		row,col = 480,840
		for i in range(row):
			for j in range(col):
				pallete[i,j,:] = colormap[id_to_cat[masks[i,j]]]
		cv2.imwrite(os.path.join(pallete_folder,img),pallete)
		colored = cv2.imread(os.path.join('../dataset/rawdata/groundtruthimage',img))
		edge = cv2.Canny(colored,0,255)
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
				color = (0,0,255)
				if cat == 1:
					color = (255,0,0)
				cv2.putText(colored,category.id_to_cat[cat],(ymin,xmin),cv2.FONT_HERSHEY_SIMPLEX,0.3,color,1,cv2.LINE_AA)
				cv2.rectangle(colored,(ymin,xmin),(ymax,xmax),(0,0,255),1)

		cv2.imwrite(os.path.join(outdir,img),colored)
# props = regionprops(labels)
# print (type(props))
# for prop in props:
# 	print(prop['label']) # individual properties can be accessed via square brackets
# 	cropped_shape = prop['filled_image'] # this gives you the content of the bounding box as an array of bool.
# 	cropped_shape = 1 * cropped_shape # convert to integer
# 	print (prop)