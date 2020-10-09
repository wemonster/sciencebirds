from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

#from ssd.SSD import _SSD
__all__ = ['DeepLabV3', 'get_deeplab']

class DeepLabV3(BaseNet):
	def __init__(self,ratio,nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
		super(DeepLabV3, self).__init__(ratio,nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

		self.head = DeepLabV3Head(2048, nclass, norm_layer, self._up_kwargs)

		self.low_level_1 = nn.Sequential( #2x
			nn.Conv2d(128,32,1,bias=False),
			norm_layer(32),
			nn.ReLU(True)
			)

		self.low_level_2 = nn.Sequential( #4x
			nn.Conv2d(256,48,1,bias=False),
			norm_layer(48),
			nn.ReLU(True)
			)
		
		self.concat_conv_1 = nn.Sequential(
			nn.Conv2d(304,128,kernel_size=3,stride=1,padding=1,bias=False),
			norm_layer(128),
			nn.ReLU(True),
			nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,bias=False),
			norm_layer(128),
			nn.ReLU(True),
			
			)

		self.concat_conv_2 = nn.Sequential(
			nn.Conv2d(160,64,kernel_size=3,stride=1,padding=1,bias=False),
			norm_layer(64),
			nn.ReLU(True),
			nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1,bias=False),
			norm_layer(32),
			nn.Conv2d(32,nclass,kernel_size=1,stride=1)
			)
		self.edge_conv = nn.Sequential(
			# nn.Conv2d(160,64,kernel_size=3,stride=1,padding=1,bias=False),
			# norm_layer(64),
			# nn.ReLU(True),
			# nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1,bias=False),
			# norm_layer(64),
			nn.Conv2d(160,2,kernel_size=3,stride=1,padding=1,bias=False)
			)

		self.objectness = nn.Sequential(
			# nn.Conv2d(160,64,kernel_size=3,stride=1,padding=1,bias=False),
			# norm_layer(64),
			# nn.ReLU(True),
			#nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1,bias=False),
			#norm_layer(64),
			nn.Conv2d(160,2,kernel_size=3,stride=1,padding=1,bias=False)
			) #foreground or background
		if aux:
			self.auxlayer = FCNHead(1024, nclass, norm_layer)

	def forward(self, x,im_info,gt_boxes,num_boxes):
		#Space for decoder
		_, _, h, w = x.size()
		c0,c1, c2, c3, c4 = self.base_forward(x)
#		print (c0.size())
		#detection head
		feature_maps = [c0,c1,c4]


		#decoder
		low_level_features1 = self.low_level_1(c0) #2x
		low_level_features2 = self.low_level_2(c1) #4x
		#print (low_level_features1.shape,low_level_features2.shape)
		x = self.head(c4)
		#print (x.shape)
		x = F.interpolate(x,(h//4,w//4),**self._up_kwargs)
		
		concated = torch.cat((low_level_features2,x),1)
		#print (concated.shape)
		# objects = F.interpolate(concated,(h,w),**self._up_kwargs)

		concated = self.concat_conv_1(concated)
		#print (concated.shape)
		x = F.interpolate(concated, (h//2,w//2), **self._up_kwargs)

		concated = torch.cat((low_level_features1,x),1)

		#print (concated.shape)
		object_edge = F.interpolate(concated,(h,w),**self._up_kwargs)

		# #print (object_edge.shape)
		# concated = self.concat_conv_2(concated)


		# x = F.interpolate(concated,(h,w), **self._up_kwargs)

		x = self.concat_conv_2(object_edge)
		edge = self.edge_conv(object_edge)
		objectness_score = self.objectness(object_edge) #batch_size x 2 x H x W

		# rois,rpn_loss_cls,rpn_loss_box = self._SSD(feature_maps,im_info,gt_boxes,num_boxes)
		return objectness_score,x,edge

	def val_forward(self,x):
		_, _, h, w = x.size()
		c0,c1, c2, c3, c4 = self.base_forward(x)

		low_level_features1 = self.low_level_1(c0)
		low_level_features2 = self.low_level_2(c1)

		x = self.head(c4)

		x = F.interpolate(x,(h//4,w//4),**self._up_kwargs)

		concated = torch.cat((low_level_features2,x),1)

		# feature_vectors = F.interpolate(concated,(h,w),**self._up_kwargs)
		concated = self.concat_conv_1(concated)

		x = F.interpolate(concated, (h//2,w//2), **self._up_kwargs)

		concated = torch.cat((low_level_features1,x),1)

		object_edge = F.interpolate(concated,(h,w),**self._up_kwargs)
		# concated = self.concat_conv_2(concated)

		# x = F.interpolate(concated,(h,w),**self._up_kwargs)
		x = self.concat_conv_2(object_edge)
		edge = self.edge_conv(object_edge)
		objectness_score = self.objectness(object_edge) #whether the pixel is fg/bg, batch_size x 2 x H x W
		return x,objectness_score,object_edge,edge

class DeepLabV3Head(nn.Module):
	def __init__(self, in_channels, out_channels, norm_layer, up_kwargs, atrous_rates=(12, 24, 36)):
		super(DeepLabV3Head, self).__init__()
		inter_channels = in_channels // 8
		self.aspp = ASPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
		self.block = nn.Sequential(
			nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
			norm_layer(inter_channels),
			nn.ReLU(True),
			nn.Dropout2d(0.1, False),
			nn.Conv2d(inter_channels, out_channels, 1))

	def forward(self, x):
		x = self.aspp(x)
		# x = self.block(x)
		return x


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
	block = nn.Sequential(
		nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
				  dilation=atrous_rate, bias=False),
		norm_layer(out_channels),
		nn.ReLU(True))
	return block

class AsppPooling(nn.Module):
	def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
		super(AsppPooling, self).__init__()
		self._up_kwargs = up_kwargs
		self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
								 nn.Conv2d(in_channels, out_channels, 1, bias=False),
								 norm_layer(out_channels),
								 nn.ReLU(True))

	def forward(self, x):
		_, _, h, w = x.size()
		pool = self.gap(x)

		return F.interpolate(pool, (h,w), **self._up_kwargs)

class ASPP_Module(nn.Module):
	def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
		super(ASPP_Module, self).__init__()
		out_channels = in_channels // 8
		rate1, rate2, rate3 = tuple(atrous_rates)
		self.b0 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 1, bias=False),
			norm_layer(out_channels),
			nn.ReLU(True))
		self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
		self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
		self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
		self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

		self.project = nn.Sequential(
			nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
			norm_layer(out_channels),
			nn.ReLU(True),
			nn.Dropout2d(0.5, False)
			)

	def forward(self, x):
		feat0 = self.b0(x)
		feat1 = self.b1(x)
		feat2 = self.b2(x)
		feat3 = self.b3(x)
		feat4 = self.b4(x)

		y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)

		return self.project(y)


def get_deeplab(ratio,nclass,dataset='sciencebirds', backbone='resnet50', pretrained=False,
				root='~/.encoding/models', **kwargs):
	# infer number of classes
	from ..datasets import datasets
	model = DeepLabV3(ratio,nclass, backbone=backbone, root=root, **kwargs)
	if pretrained:
		raise NotImplementedError

	return model
