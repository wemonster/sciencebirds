





import os
import cv2
import random
import numpy as np
from enum import Enum
import copy

class GameObjectType(Enum):
	BLACKBIRD = 'BLACKBIRD'
	BLUEBIRD = 'BLUEBIRD'
	HILL = 'HILL'
	ICE = 'ICE'
	PIG = 'PIG'
	REDBIRD = 'REDBIRD'
	STONE = 'STONE'
	WHITEBIRD = 'WHITEBIRD'
	WOOD = 'WOOD'
	YELLOWBIRD = 'YELLOWBIRD'
	SLING = 'SLING'
	TNT = 'TNT'
	BACKGROUND = 'BACKGROUND'

gameObjectType = {
	'BACKGROUND':0,
	'BLACKBIRD':1,
	'BLUEBIRD':2,
	'HILL':3,
	'ICE':4,
	'PIG':5,
	'REDBIRD':6,
	'STONE':7,
	'WHITEBIRD':8,
	'WOOD':9,
	'YELLOWBIRD':10,
	'SLING':11,
	'TNT':12,
}

id_to_cat = {
	0:'BACKGROUND',
	1:'BLACKBIRD',
	2:'BLUEBIRD',
	3:'HILL',
	4:'ICE',
	5:'PIG',
	6:'REDBIRD',
	7:'STONE',
	8:'WHITEBIRD',
	9:'WOOD',
	10:'YELLOWBIRD',
	11:'SLING',
	12:'TNT'
}


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
	'TNT':[64,128,128]
}


all_categories = ['BLACKBIRD','BLUEBIRD','HILL','ICE','PIG','REDBIRD','STONE','WHITEBIRD',
		'WOOD','YELLOWBIRD','SLING','HILL','TNT']
# colormap = {}

# for i in range(len(all_categories)):
	# colormap[cat] = [i,i,i]
gts_root = "dataset/rawdata/groundtruth"
gts = sorted(os.listdir(gts_root),key=lambda x:int(x.split('.')[0][11:]))
gtimage_root = "dataset/rawdata/groundtruthimage"
gtimages = sorted(os.listdir(gtimage_root),key=lambda x:int(x.split('.')[0]))
print (len(gts),len(gtimages))
for j in range(1):
	truth = open(os.path.join(gts_root,gts[j]),'r').readlines()
	im = cv2.imread(os.path.join(gtimage_root,gtimages[j]))
	cv2.imwrite("dataset/images/{}".format(gtimages[j]),im)
	i = 0
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,binary = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)
	mask = np.zeros((480,840,3)).astype(np.uint8)
	label = np.zeros((480,840)).astype(np.uint8)
	for t in truth:
		info = t.split('|')
		X = int(info[0])
		Y = int(info[1])
		height = int(info[2])
		width = int(info[3])
		vertices = info[4]
		game_type = str(info[5]).strip().split('.')[1]
		if game_type == 'UNKNOWN':
			continue
		startPoint = (X,Y)
		endPoint = (X + height,Y+width)
		to_ret = np.zeros((480,840)).astype(np.uint8)
		# cv2.rectangle(im,startPoint,endPoint,(255,0,0),1)
		to_ret[Y:Y+width,X:X+height] = binary[Y:Y+width,X:X+height]
		contours,hierarchy = cv2.findContours(to_ret,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		cv2.fillPoly(to_ret[Y:Y+width,X:X+height],contours,gameObjectType[game_type],1)
		temp = to_ret[Y:Y+width,X:X+height]
		temp[temp>0] = gameObjectType[game_type]
		temp = gameObjectType[game_type] - temp
		#label
		label[Y:Y+width,X:X+height] = temp
		row,col = temp.shape
		temp3d = np.zeros((row,col,3)).astype(np.uint8)
		for m in range(row):
			for n in range(col):
				if temp[m,n] == 0:
					continue
				target_colormap = colormap[id_to_cat[temp[m,n]]]
				temp3d[m,n,:] = copy.deepcopy(target_colormap)
		mask[Y:Y+width,X:X+height,:] = temp3d
	print(label.shape)
	# cv2.imwrite("dataset/masks/{}".format(gtimages[j]),mask)
	# cv2.imwrite('dataset/labels/{}'.format(gtimages[j]),label)
	print ("finish writing images {}".format(j))
	

def generate_imagesets(ratio):
	annotation_folder = "dataset/images"
	files = os.listdir(annotation_folder)
	# print (files)
	filename = [str(int(x.split('.')[0]))+'\n' for x in files]
	train_file = open('dataset/ImageSets/train.txt','w')
	val_file = open('dataset/ImageSets/val.txt','w')

	for i in filename:
		if int(i) >= 2400 and int(i) <= 2425:
			continue
		prob = random.random()
		if prob <= ratio:
			train_file.write("{}".format(i))
		else:
			val_file.write('{}'.format(i))
	train_file.close()
	val_file.close()

# generate_imagesets(0.8)