import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
	def __init__(self, phase, size, base, extras, head, num_classes):
		super(SSD, self).__init__()
		self.phase = phase
		self.num_classes = num_classes
		self.cfg = (coco, voc)[num_classes == 21]
		self.priorbox = PriorBox(self.cfg)
		self.priors = Variable(self.priorbox.forward(), volatile=True)
		self.size = size

		# SSD network
		self.vgg = nn.ModuleList(base)
		# Layer learns to scale the l2 normalized features from conv4_3
		self.L2Norm = L2Norm(512, 20)
		self.extras = nn.ModuleList(extras)

		self.loc = nn.ModuleList(head[0])
		self.conf = nn.ModuleList(head[1])

		if phase == 'test':
			self.softmax = nn.Softmax(dim=-1)
			self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

	def forward(self,feature_maps):

		sources = list() #output of the feature maps.
		loc = list() #location of bounding box
		conf = list() #category

