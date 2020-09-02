

import os
import cv2
import random
gts_root = "groundtruth"
gts = sorted(os.listdir(gts_root),key=lambda x:int(x.split('.')[0][11:]))
gtimage_root = "groundtruthimage"
gtimages = sorted(os.listdir(gtimage_root),key=lambda x:int(x.split('.')[0]))
print (len(gts),len(gtimages))
for j in range(10 ):
	truth = open(os.path.join(gts_root,gts[j]),'r').readlines()
	im = cv2.imread(os.path.join(gtimage_root,gtimages[j]))
	for t in truth:
		info = t.split('|')
		X = int(info[0])
		Y = int(info[1])
		height = int(info[2])
		width = int(info[3])
		vertices = info[4]
		game_type = str(info[5]).strip()
		startPoint = (X,Y)
		endPoint = (X + height,Y+width)
		cv2.rectangle(im,startPoint,endPoint,(255,0,0),1)
		print (game_type)
	# cv2.imwrite("labelimage/{}".format(gtimages[i]),im)
	cv2.imshow('1',im)
	cv2.waitKey(0)

