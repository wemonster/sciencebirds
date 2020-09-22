





import os
import cv2
import random
import numpy as np
from enum import Enum
import copy

from classlabel import Category


all_categories = ['BLACKBIRD','BLUEBIRD','HILL','ICE','PIG','REDBIRD','STONE','WHITEBIRD',
		'WOOD','YELLOWBIRD','SLING','HILL','TNT']
# colormap = {}

# for i in range(len(all_categories)):
	# colormap[cat] = [i,i,i]
def generate_dataset(classes,ratio):
	classids = Category(classes,True)
	gts_root = "dataset/rawdata/groundtruth"
	gts = sorted(os.listdir(gts_root),key=lambda x:int(x.split('.')[0][11:]))
	gtimage_root = "dataset/rawdata/groundtruthimage"
	gtimages = sorted(os.listdir(gtimage_root),key=lambda x:int(x.split('.')[0]))
	print (len(gts),len(gtimages))
	save_folder = "dataset/images/small/{}".format(int(ratio*10))
	for j in range(len(gts)):
		truth = open(os.path.join(gts_root,gts[j]),'r').readlines()
		im = cv2.imread(os.path.join(gtimage_root,gtimages[j]))
		cv2.imwrite(os.path.join(save_folder,"rawimage/{}".format(gtimages[j])),im)
		i = 0
		gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		ret,binary = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)
		mask = np.zeros((480,840,3)).astype(np.uint8)
		label = np.zeros((480,840)).astype(np.uint8)
		# label slingshot first
		# for t in truth:
		# 	info = t.split('|')
		# 	X = int(info[0])
		# 	Y = int(info[1])
		# 	height = int(info[2])
		# 	width = int(info[3])
		# 	vertices = info[4]
		# 	game_type = str(info[5]).strip().split('.')[1]
		# 	if game_type not in classes:
		# 		continue
		# 		# game_type = 'UNKNOWN'
		# 	if game_type == 'SLING':

		# 		startPoint = (X,Y)
		# 		endPoint = (X + height,Y+width)
		# 		to_ret = np.zeros((480,840)).astype(np.uint8)
		# 		# cv2.rectangle(im,startPoint,endPoint,(255,0,0),1)
		# 		to_ret[Y:Y+width,X:X+height] = binary[Y:Y+width,X:X+height]
		# 		contours,hierarchy = cv2.findContours(to_ret,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		# 		cv2.fillPoly(to_ret[Y:Y+width,X:X+height],contours,classids.gameObjectType[game_type],1)
		# 		temp = to_ret[Y:Y+width,X:X+height]
		# 		temp[temp>0] = classids.gameObjectType[game_type]
		# 		temp = classids.gameObjectType[game_type] - temp
		# 		#label
		# 		label[Y:Y+width,X:X+height] = temp
		# 		break

		# for t in truth:
		# 	info = t.split('|')
		# 	X = int(info[0])
		# 	Y = int(info[1])
		# 	height = int(info[2])
		# 	width = int(info[3])
		# 	vertices = info[4]
		# 	game_type = str(info[5]).strip().split('.')[1]
		# 	if game_type == 'SLINGSHOT' or game_type not in classes:
		# 		continue
		# 	startPoint = (X,Y)
		# 	endPoint = (X + height,Y+width)
		# 	to_ret = np.zeros((480,840)).astype(np.uint8)
		# 	# cv2.rectangle(im,startPoint,endPoint,(255,0,0),1)
		# 	to_ret[Y:Y+width,X:X+height] = binary[Y:Y+width,X:X+height]
		# 	contours,hierarchy = cv2.findContours(to_ret,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		# 	cv2.fillPoly(to_ret[Y:Y+width,X:X+height],contours,classids.gameObjectType[game_type],1)
		# 	temp = to_ret[Y:Y+width,X:X+height]
		# 	temp[temp>0] = classids.gameObjectType[game_type]
		# 	temp = classids.gameObjectType[game_type] - temp
		# 	#label
		# 	label[Y:Y+width,X:X+height] = temp
		# 	# row,col = temp.shape
		# 	# temp3d = np.zeros((row,col,3)).astype(np.uint8)
		# 	# for m in range(row):
		# 	# 	for n in range(col):
		# 	# 		if temp[m,n] == 0:
		# 	# 			continue
		# 	# 		target_colormap = classids.colormap[classids.id_to_cat[temp[m,n]]]
		# 	# 		temp3d[m,n,:] = copy.deepcopy(target_colormap)
		# 	# mask[Y:Y+width,X:X+height,:] = temp3d
		# cv2.imwrite(os.path.join(save_folder,"masks/{}".format(gtimages[j])),label)
		# label[label>0] = 1
		# foreground = np.multiply(im,label[:,:,np.newaxis])
		# # cv2.imwrite(os.path.join("dataset/rawdata/foregrounds","{}".format(gtimages[j])),foreground)
		# cv2.imwrite(os.path.join(save_folder,'foregrounds/{}'.format(gtimages[j])),foreground)
		# #label for unknowns
		classids = Category(classes,True)
		label = np.zeros((480,840)).astype(np.uint8)
		for t in truth:
			info = t.split('|')
			X = int(info[0])
			Y = int(info[1])
			height = int(info[2])
			width = int(info[3])
			vertices = info[4]
			game_type = str(info[5]).strip().split('.')[1]
			if game_type not in classes:
				continue
			if game_type == 'SLING':
				startPoint = (X,Y)
				endPoint = (X + height,Y+width)
				to_ret = np.zeros((480,840)).astype(np.uint8)
				# cv2.rectangle(im,startPoint,endPoint,(255,0,0),1)
				to_ret[Y:Y+width,X:X+height] = binary[Y:Y+width,X:X+height]
				contours,hierarchy = cv2.findContours(to_ret,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
				cv2.fillPoly(to_ret[Y:Y+width,X:X+height],contours,classids.gameObjectType[game_type],1)
				temp = to_ret[Y:Y+width,X:X+height]
				temp[temp>0] = classids.gameObjectType[game_type]
				temp = classids.gameObjectType[game_type] - temp
				#label
				label[Y:Y+width,X:X+height] = temp
				break
		for t in truth:
			info = t.split('|')
			X = int(info[0])
			Y = int(info[1])
			height = int(info[2])
			width = int(info[3])
			vertices = info[4]
			game_type = str(info[5]).strip().split('.')[1]
			if game_type == 'SLING':
				continue
			if game_type not in classes:
				game_type = 'UNKNOWN'
			startPoint = (X,Y)
			endPoint = (X + height,Y+width)
			to_ret = np.zeros((480,840)).astype(np.uint8)
			# cv2.rectangle(im,startPoint,endPoint,(255,0,0),1)
			to_ret[Y:Y+width,X:X+height] = binary[Y:Y+width,X:X+height]
			contours,hierarchy = cv2.findContours(to_ret,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
			cv2.fillPoly(to_ret[Y:Y+width,X:X+height],contours,classids.gameObjectType[game_type],1)
			temp = to_ret[Y:Y+width,X:X+height]
			temp[temp>0] = classids.gameObjectType[game_type]
			temp = classids.gameObjectType[game_type] - temp
			#label
			label[Y:Y+width,X:X+height] = temp
		label[label>0] = 1
		foreground = np.multiply(im,label[:,:,np.newaxis])
		cv2.imwrite(os.path.join("dataset/rawdata",'foregrounds/{}'.format(gtimages[j])),foreground)
		# cv2.imwrite(os.path.join(save_folder,"unknowns/{}".format(gtimages[j])),label)
		print ("finish writing images {}".format(j))
	

def generate_imagesets(ratio):
	annotation_folder = "dataset/rawdata/groundtruthimage"
	files = os.listdir(annotation_folder)
	# print (files)
	filename = [str(int(x.split('.')[0]))+'\n' for x in files]
	train_file = open('dataset/ImageSets/train.txt','w')
	val_file = open('dataset/ImageSets/val.txt','w')
	test_file = open('dataset/ImageSets/test.txt','w')
	for i in filename:
		if int(i) >= 2400 and int(i) <= 2425:
			continue
		prob = random.random()
		if prob <= ratio:
			train_file.write("{}".format(i))
		elif prob > ratio and prob < ratio+0.1:
			val_file.write('{}'.format(i))
		else:
			test_file.write('{}'.format(i))
	train_file.close()
	val_file.close()
	test_file.close()

# data = open("logs/resnet.txt",'r').readlines()
# # class_info = []
# for i in data[:1]:
# 	ratio,classes = i.split('|')
# 	ratio = float(ratio.split(':')[1])
# 	classes = classes.strip().split(':')[1].split(',')
# 	generate_dataset(classes,ratio)
# generate_imagesets(0.8)

a = cv2.imread("dataset/rawdata/foregrounds/0.png")

cv2.ellipse(a,(256,256),(100,50),0,0,255,(255,255,255),-1)
cv2.rectangle(a,(30,30),(100,100),(255,255,255),40)
cv2.imwrite("dataset/rawdata/foregrounds/4400.png",a)