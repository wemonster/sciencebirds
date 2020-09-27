

import numpy as np

import skimage.measure
from skimage.measure import regionprops
import cv2
# masks = cv2.imread("dataset/images/small/0/masks/0.png",0)
masks = cv2

colored = cv2.imread('dataset/rawdata/groundtruthimage/0.png')
categories = np.unique(masks)
for cat in categories:
	if cat == 0:
		continue
	to_ret = np.zeros((480,840)).astype(np.uint8)
	to_ret[(masks==cat)] = 255
	labels,num = skimage.measure.label(to_ret,connectivity=2,return_num=True)
	props = regionprops(labels)
	print (cat,num)
	for prop in props:
		print (prop['bbox'])
		xmin,ymin,xmax,ymax = prop['bbox']
		cv2.rectangle(colored,(ymin,xmin),(ymax,xmax),(255,0,0),1)

cv2.imwrite("bounded_box.png",colored)
# props = regionprops(labels)
# print (type(props))
# for prop in props:
# 	print(prop['label']) # individual properties can be accessed via square brackets
# 	cropped_shape = prop['filled_image'] # this gives you the content of the bounding box as an array of bool.
# 	cropped_shape = 1 * cropped_shape # convert to integer
# 	print (prop)