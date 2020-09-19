from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['DeepLabV3', 'get_deeplab']

class DeepLabV3(BaseNet):
	def __init__(self,ratio,nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
		super(DeepLabV3, self).__init__(ratio,nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

		self.head = DeepLabV3Head(2048, nclass, norm_layer, self._up_kwargs)

		self.low_level = nn.Sequential(
			nn.Conv2d(256,48,1,bias=False),
			norm_layer(48),
			nn.ReLU(True)
			)
		
		self.concat_conv = nn.Sequential(
			nn.Conv2d(304,256,kernel_size=3,stride=1,padding=1,bias=False),
			norm_layer(256),
			nn.ReLU(True),
			nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=False),
			norm_layer(256),
			nn.ReLU(True),
			nn.Conv2d(256,nclass,kernel_size=1,stride=1)
			)
		
		self.objectness = nn.Conv2d(304,2,kernel_size=1,stride=1,padding=1,bias=False) #foreground or background
		if aux:
			self.auxlayer = FCNHead(1024, nclass, norm_layer)

	def forward(self, x):
		#Space for decoder
		_, _, h, w = x.size()
		c1, c2, c3, c4 = self.base_forward(x)


		low_level_features = self.low_level(c1)

		x = self.head(c4)

		x = F.interpolate(x,(h//4,w//4),**self._up_kwargs)

		concated = torch.cat((low_level_features,x),1)
		objects = F.interpolate(concated,(h,w),**self._up_kwargs)
		concated = self.concat_conv(concated)
		x = F.interpolate(concated, (h,w), **self._up_kwargs)
		# labeled = F.softmax(x,dim=1)
		labeled = x
		objectness_score = self.objectness(objects) #batch_size x 2 x H x W
		# print (torch.sum(x,dim=1))
		# labeled = torch.argmax(labeled,dim=1)
		# outputs.append(x)
		# if self.aux:
		#     auxout = self.auxlayer(c3)
		#     auxout = F.interpolate(auxout, (h,w), **self._up_kwargs)
		#     outputs.append(auxout)

		# return tuple(outputs)
		# print (x)
		return objectness_score,labeled

	def val_forward(self,x):
		_, _, h, w = x.size()
		c1, c2, c3, c4 = self.base_forward(x)


		low_level_features = self.low_level(c1)

		x = self.head(c4)

		x = F.interpolate(x,(h//4,w//4),**self._up_kwargs)

		concated = torch.cat((low_level_features,x),1)
		feature_vectors = F.interpolate(concated,(h,w),**self._up_kwargs)
		concated = self.concat_conv(concated)

		x = F.interpolate(concated, (h,w), **self._up_kwargs)
		objectness_score = self.objectness(feature_vectors)
		return x,objectness_score,feature_vectors

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
