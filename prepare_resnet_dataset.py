


import cv2
import os
import numpy as np
import random
'''
	Shapes: rectangle, circle, ellipse, star, polygons, nonagon, right,rhombus,parallelogram...
	Colors: (0~255,0~255,0~255)
	'''
def write_squares(num_samples,ratio=0.8):
	rect = "rect"
	for i in range(num_samples):
		#choose size
		size = random.choice(sizes)
		m,n = size

		img = np.zeros((m,n,3),np.uint8)

		#choose padding
		padding = random.choice(paddings)
		startpoint = (padding,padding)
		endpoint = (m-padding,n-padding)

		#choose colour
		color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

		#choose filled
		filled = random.choice(fills)

		cv2.rectangle(img,startpoint,endpoint,color,filled)

		prob = random.random()
		path = "dataset/geometric"
		if prob <= ratio:
			path = os.path.join(path,"train/square")
		elif prob > ratio and prob < (1-ratio)/2 + ratio:
			path = os.path.join(path,"val/square")
		else:
			path = os.path.join(path,"test/square")
		written_name = "{}{}.png".format(rect,i)
		cv2.imwrite(os.path.join(path,written_name),img)

#prepare circles
def write_circles(num_samples,ratio=0.8):
	circ = "circ"
	for i in range(num_samples):
		#choose size
		size = random.choice(sizes)
		m,n = size

		img = np.zeros((m,n,3),np.uint8)

		#choose padding
		padding = random.choice(paddings)
		center = (m // 2, n // 2)
		radius = m // 2 - padding

		color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

		filled = random.choice(fills)
		cv2.circle(img,center,radius,color,filled)

		prob = random.random()
		path = "dataset/geometric"
		if prob <= ratio:
			path = os.path.join(path,"train/circle")
		elif prob > ratio and prob < (1-ratio)/2 + ratio:
			path = os.path.join(path,"val/circle")
		else:
			path = os.path.join(path,"test/circle")
		written_name = "{}{}.png".format(circ,i)

		cv2.imwrite(os.path.join(rawdata,written_name),img)

#prepare ellipses
def write_ellipses(num_samples,ratio=0.8):
	ellip = "ellip"
	for i in range(num_samples):
		#choose size
		size = random.choice(sizes)
		m,n = size

		img = np.zeros((m,n,3),np.uint8)

		#choose padding
		padding = random.choice(paddings)

		center = (m // 2, n // 2)

		axesLength = (random.randint(padding,n//2),random.randint(padding,m//2))
		angle = 0
		startAngle = 0
		endAngle = random.randint(90,360)
		color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
		filled = random.choice(fills)
		cv2.ellipse(img,center,axesLength,angle,startAngle,endAngle,color,filled)

		written_name = "{}{}.png".format(ellip,i)
		cv2.imwrite(os.path.join(rawdata,written_name),img)

#prepare triangle
def write_trigs(num_samples,ratio=0.8):
	trig = "trig"
	for i in range(num_samples):
		#choose size
		size = random.choice(sizes)
		m,n = size

		img = np.zeros((m,n,3),np.uint8)

		#choose padding
		padding = random.choice(paddings)

		center = (m // 2, n // 2)

		#separate the img into three regions
		point1 = [random.randint(padding,n),random.randint(padding,m//2)]
		point2 = [random.randint(padding,n//2),random.randint(m//2+padding,m)]
		point3 = [random.randint(n//2+padding,n),random.randint(m//2+padding,m)]
		points = np.array([point1,point2,point3],np.int32).reshape((-1,1,2))

		color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
		filled = random.choice(fills)
		cv2.polylines(img,[points],True,color,1,lineType=cv2.LINE_AA)
		
		written_name = "{}{}.png".format(trig,i)
		cv2.imwrite(os.path.join(rawdata,written_name),img)

		cv2.fillPoly(img,[points],color)
		written_name = "{}_fill{}.png".format(trig,i)
		cv2.imwrite(os.path.join(rawdata,written_name),img)

#prepare right triangles
def write_right_trigs(num_samples,ratio=0.8):
	righttrig = "right_trig"
	for i in range(num_samples):
		#choose size
		size = random.choice(sizes)
		m,n = size

		img = np.zeros((m,n,3),np.uint8)

		#choose padding
		padding = random.choice(paddings)

		center = (m // 2, n // 2)

		#separate the img into three regions
		#col x row
		point1 = [random.randint(padding,n//2),random.randint(padding,m//2)] #top
		point2 = [point1[0],random.randint(m//2+padding,m)] #bottom left
		point3 = [random.randint(n//2+padding,n),point2[1]] #bottom right
		points = np.array([point1,point2,point3],np.int32).reshape((-1,1,2))

		color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
		cv2.polylines(img,[points],True,color,1,lineType=cv2.LINE_AA)
		
		written_name = "{}{}.png".format(righttrig,i)
		cv2.imwrite(os.path.join(rawdata,written_name),img)

		cv2.fillPoly(img,[points],color)
		written_name = "{}_fill{}.png".format(righttrig,i)
		cv2.imwrite(os.path.join(rawdata,written_name),img)

#prepare left triangles
def write_left_trigs(num_samples,ratio=0.8):
	left_trig = "left_trig"
	for i in range(num_samples):
		#choose size
		size = random.choice(sizes)
		m,n = size

		img = np.zeros((m,n,3),np.uint8)

		#choose padding
		padding = random.choice(paddings)

		center = (m // 2, n // 2)

		#separate the img into three regions
		#col x row
		point1 = [random.randint(n//2+padding,n),random.randint(padding,m//2)] #top
		point3 = [point1[0],random.randint(n//2+padding,n)] #bottom right
		point2 = [random.randint(padding,n//2),point3[1]] #bottom left
		
		points = np.array([point1,point2,point3],np.int32).reshape((-1,1,2))

		color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
		cv2.polylines(img,[points],True,color,1,lineType=cv2.LINE_AA)
		
		written_name = "{}{}.png".format(left_trig,i)
		cv2.imwrite(os.path.join(rawdata,written_name),img)

		cv2.fillPoly(img,[points],color)
		written_name = "{}_fill{}.png".format(left_trig,i)
		cv2.imwrite(os.path.join(rawdata,written_name),img)

#prepare star
def write_stars(num_samples):
	star = "star"
	for i in range(num_samples):
		#choose size
		size = random.choice(sizes)
		m,n = size

		img = np.zeros((m,n,3),np.uint8)

		#choose padding
		padding = random.choice(paddings)

		center = (m // 2, n // 2)

		#separate the img into three regions
		#col x row
		point1 = [random.randint(n//2+padding,n),random.randint(padding,m//2)] #top
		point3 = [point1[0],random.randint(n//2+padding,n)] #bottom right
		point2 = [random.randint(padding,n//2),point3[1]] #bottom left
		
		points = np.array([point1,point2,point3],np.int32).reshape((-1,1,2))

		color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
		cv2.polylines(img,[points],True,color,1,lineType=cv2.LINE_AA)
		
		written_name = "{}{}.png".format(star,i)
		cv2.imwrite(os.path.join(rawdata,written_name),img)

		cv2.fillPoly(img,[points],color)
		written_name = "{}_fill{}.png".format(star,i)
		cv2.imwrite(os.path.join(rawdata,written_name),img)
#prepare nonagon

#prepare parallelogram

#prepare trapezium


#prepare random polygons

#fill all of them
if __name__ == "__main__":
	rawdata = "dataset/rawdata/shapes"
	
	shapes = ['square','circle','ellipse','right_trig','left_trig']
	phases = ['train','val','test']
	for i in shapes:
		for phase in phases:
			if not os.path.exists("dataset/geometric/{}/{}".format(phase,i)):
				os.mkdir("dataset/geometric/{}/{}".format(phase,i))
	num_samples = 1000
	sizes = [(32,32),(64,64),(128,128),(224,224)]
	paddings = [5,10]
	fills = [1,-1]
	ratio = 0.8
	write_squares(num_samples)
	write_circles(num_samples)
	write_ellipses(num_samples)
	write_right_trigs(num_samples)
	write_left_trigs(num_samples)
	# write_stars(num_samples)
	# write_non