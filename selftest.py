


import cv2
import numpy as np

foreground = cv2.imread("../sciencebirds_dataset/rawdata/foregrounds/0.png")

objectness = cv2.cvtColor(foreground,cv2.COLOR_BGR2GRAY)
# objectness = np.array(foreground)

objectness[objectness > 0] = 255
# print (np.unique(objectness))
cv2.imwrite("foreground.png",objectness)