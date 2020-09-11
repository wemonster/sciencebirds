###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import numpy as np
from tqdm import tqdm
import copy
import cv2
import random
from PIL import Image


import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model

from option import Options
from maskimage import generate_dataset

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
	from torch.autograd import Variable

class Trainer():
	def __init__(self,info,id_info,args):
		self.ratio,self.classes = info[0],info[1]
		self.categories = id_info
		self.args = args
		# data transforms
		input_transform = transform.Compose([
			transform.ToTensor(),
			transform.Normalize([.485, .456, .406], [.229, .224, .225])
			])
		label_transform = transform.ToTensor()
		# dataset
		data_kwargs = {'transform': input_transform, 'target_transform':input_transform,
						'label_transform':label_transform,
						 'base_size': args.base_size,'crop_size': args.crop_size}
		trainset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train',
										   **data_kwargs)

		testset = get_segmentation_dataset(args.dataset, split='val', mode ='val',
										   **data_kwargs)
		print ("finish loading the dataset")
		# dataloader
		kwargs = {'num_workers': args.workers, 'pin_memory': True} \
			if args.cuda else {}
		self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
										   drop_last=True, shuffle=True, **kwargs)
		self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
										 drop_last=False, shuffle=False, **kwargs)
		self.nclass = int((1-self.ratio) * 12) + 2
		# model
		model = get_segmentation_model(self.ratio,self.nclass,args.model, dataset = args.dataset,
									   backbone = args.backbone, dilated = args.dilated,
									   lateral = args.lateral, jpu = args.jpu, aux = args.aux,
									   se_loss = args.se_loss, #norm_layer = SyncBatchNorm,
									   base_size = args.base_size, crop_size = args.crop_size)

		# print(model)
		# for (name,w) in model.named_parameters():
		# 	print (name,w.requires_grad)
		# optimizer using different LR
		params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr}]
		params_list.append({'params':model.low_level.parameters(),'lr':args.lr})
		params_list.append({'params':model.concat_conv.parameters(),'lr':args.lr})
		# if hasattr(model, 'jpu'):
		# 	params_list.append({'params': model.jpu.parameters(), 'lr': args.lr*10})
		if hasattr(model, 'head'): 
			params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
		# if hasattr(model, 'auxlayer'):
		# 	params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
		optimizer = torch.optim.SGD(params_list, lr=args.lr,
			momentum=args.momentum, weight_decay=args.weight_decay)
		# criterions
		self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux,
											nclass=self.nclass, 
											se_weight=args.se_weight,
											aux_weight=args.aux_weight)
		self.model, self.optimizer = model, optimizer
		# using cuda
		if args.cuda:
			self.model = DataParallelModel(self.model).cuda()
			self.criterion = DataParallelCriterion(self.criterion).cuda()
		# resuming checkpoint
		self.best_pred = 0.0
		if args.resume is not None:
			if not os.path.isfile(args.resume):
				raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			if args.cuda:
				self.model.module.load_state_dict(checkpoint['state_dict'])
			else:
				self.model.load_state_dict(checkpoint['state_dict'])
			if not args.ft:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.best_pred = checkpoint['best_pred']
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		# clear start epoch if fine-tuning
		if args.ft:
			args.start_epoch = 0
		# lr scheduler
		self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
											args.epochs, len(self.trainloader))

		self.correct_features = torch.tensor([])
		self.corresponding_class = torch.tensor([])

	def training(self, epoch,log_file):

		training_log = open("logs/train/{}.txt".format(epoch),'w')
		train_loss = 0.0
		self.model.train()
		tbar = tqdm(self.trainloader)
		for i, (image,labels) in enumerate(tbar):
			self.scheduler(self.optimizer, i, epoch, self.best_pred)
			self.optimizer.zero_grad()
			if torch_ver == "0.3":
				image = Variable(image)
				labels = Variable(labels)
			outputs,labeled = self.model(image)
			labeled = labeled.type(torch.cuda.FloatTensor)
			
			labels = torch.squeeze(labels)
			labels = labels.to(dtype=torch.int64)

			loss = self.criterion(labeled, labels)
			loss.backward()
			self.optimizer.step()
			train_loss += loss.item()
			tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
			training_log.write("Iteration:{}, Loss:{:.3f}\n".format(i,train_loss/(i+1)))
		log_file.write("Epoch:{}, Loss:{:.3f}\n".format(epoch,train_loss/(i+1)))
		training_log.close()
		if self.args.no_val:
			# save checkpoint every epoch
			is_best = False
			utils.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': self.model.module.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'best_pred': self.best_pred,
			}, self.args, is_best, filename='checkpoint_{}.pth.tar'.format(epoch))


	def validation(self, epoch,log_file):
		# Fast test during the training
		val_log = open("logs/val/{}.txt".format(epoch),'w')
		def collect_features(features,position,pred):
			'''
			features: batch_size x 304 x H x W
			position: (tuple1,tuple2,tuple3); tuple1 = img_id, tuple2 = x, tuple3 = y
			
			return: 304 x (#correctly classified)
			'''
			if len(position) == 3:
				img = position[0]
				x = position[1]
				y = position[2]

				#random sampling for 100 samples during each validation
				number_matched = len(img)
				chosen = 100
				random_samples = random.sample(list(range(number_matched)),min(number_matched,chosen))
				img = img[random_samples]
				x = x[random_samples]
				y = y[random_samples]
				if len(self.corresponding_class) == 0:
					self.corresponding_class = pred[img,x,y]
				else:
					self.corresponding_class = torch.cat((self.corresponding_class,pred[img,x,y]))
				if len(self.correct_features) == 0:
					self.correct_features = features[img,:,x,y]
				else:
					self.correct_features = torch.cat((self.correct_features,features[img,:,x,y]))
				print (self.correct_features.size(),self.corresponding_class.size())
			else:
				#there is only 1 image in the batch
				x = position[0]
				y = position[1]
				#random sampling for 100 samples during each validation
				number_matched = len(x)
				chosen = 100
				random_samples = random.sample(list(range(number_matched)),min(number_matched,chosen))
				x = x[random_samples]
				y = y[random_samples]
				if len(self.corresponding_class) == 0:
					self.corresponding_class = pred[x,y]
				else:
					self.corresponding_class = torch.cat((self.corresponding_class,pred[x,y]))
				if len(self.correct_features) == 0:
					self.correct_features = features[:,x,y]
				else:
					self.correct_features = torch.cat((self.correct_features,features[:,x,y]))
				print (self.correct_features.size(),self.corresponding_class.size())


		def eval_batch(model, image, target):
			labeled,features = model.module.val_forward(image)
			pred = torch.argmax(labeled,dim=1)
			target = target.squeeze().cuda()
			correct, labeled,correct_classified = utils.batch_pix_accuracy(pred.data, target)
			collect_features(features,correct_classified,pred)
			inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
			return correct, labeled, inter, union

		

		is_best = False
		self.model.eval()
		total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
		tbar = tqdm(self.valloader, desc='\r')
		for i, (image,labels) in enumerate(tbar):
			image = image.type(torch.cuda.FloatTensor)
			if torch_ver == "0.3":
				image = Variable(image, volatile=True)
				correct, labeled, inter, union = eval_batch(self.model, image, labels)
			else:
				with torch.no_grad():
					correct, labeled, inter, union = eval_batch(self.model, image, labels)

			total_correct += correct
			total_label += labeled
			total_inter += inter
			total_union += union
			pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
			IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
			mIoU = IoU.mean()
			tbar.set_description(
				'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))
			val_log.write("Iteration:{}, pixAcc:{:.3f}, mIoU:{:.3f}\n".format(i,pixAcc,mIoU))
		val_log.close()
		new_pred = (pixAcc + mIoU)/2
		log_file.write("Epoch:{}, pixAcc:{:.3f}, mIoU:{:.3f}, Overall:{:.3f}\n".format(epoch,pixAcc,mIoU,new_pred))
		if new_pred >= self.best_pred:
			is_best = True
			self.best_pred = new_pred
			utils.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': self.model.module.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'best_pred': new_pred,
			}, self.args, is_best,self.ratio,"checkpoint_{}.pth.tar".format(epoch+1))

	def build_gaussian_model(self):
		occurrance = self.corresponding_class.cpu().numpy()
		(category,occurrance) = np.unique(occurrance,return_counts=True)
		class_mean = {}
		class_var = {}
		if not os.path.exists("../models/gaussian"):
			os.mkdir("../models/gaussian")
		for i in range(len(self.corresponding_class)):
			target_category = self.corresponding_class[i]
			if target_category not in class_mean:
				class_mean["mean_{}".format(target_category)] = self.correct_features[i,:]
			else:
				class_mean["mean_{}".format(target_category)] += self.correct_features[i,:]

		#calculate mean for each class
		for i in range(len(category)):
			class_mean["mean_{}".format(category[i])] /= occurrance[i]

		#calculate var for each class
		for i in category:
			target_id = i
			target_category = (self.corresponding_class == target_id).nonzero()
			if len(target_category) != 0: #that class exists in our
				matched_features = self.correct_features[target_category,:].cpu().numpy().squeeze(axis=1)
				class_var["cov_{}".format(target_id)] = 1/(len(target_category) - 1) * np.dot(matched_features.T,matched_features)

		torch.save(class_mean,os.path.join("../models/gaussian","mean_{}.pt".format(int(self.ratio*10))))
		torch.save(class_var,os.path.join("../models/gaussian","var_{}.pt".format(int(self.ratio*10))))

class Category:
	def __init__(self,classes):
		self.gameObjectType = {
			'BACKGROUND':0,
			'UNKNOWN':1
		}

		self.id_to_cat = {
			0:'BACKGROUND',
			1:'UNKNOWN'
		}
		self.colormap = {
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
		for i in range(len(classes)):
			self.gameObjectType[classes[i]] = i+2
			self.id_to_cat[i+2] = classes[i]

	@property
	def ids(self):
		return self.id_to_cat.keys()
	

	def convert_class_to_category(self,class_name):
		return self.gameObjectType[class_name]

	def convert_category_to_class(self,category_id):
		return self.id_to_cat[category_id]


def get_class_lists():
	data = open("logs/resnet.txt",'r').readlines()
	class_info = []
	for i in data:
		ratio,classes = i.split('|')
		ratio = float(ratio.split(':')[1])
		classes = classes.strip().split(',')
		class_info.append((ratio,classes))
	return class_info


if __name__ == "__main__":
	args = Options().parse()
	torch.manual_seed(args.seed)
	
	
	train_log_file = open("logs/training_log.txt",'w')
	val_log_file = open("logs/val_log.txt",'w')
	class_info = get_class_lists()
	print (class_info)

	for i in range(len(class_info)):
		id_info = Category(class_info[i][1])
		trainer = Trainer(class_info[i],id_info,args)
		print('Starting Epoch:', trainer.args.start_epoch)
		print('Total Epoches:', trainer.args.epochs)
		generate_dataset(class_info[i][1])
		for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
			trainer.training(info,epoch,train_log_file)
			if not trainer.args.no_val:
				trainer.validation(epoch,val_log_file)
		trainer.build_gaussian_model()
	train_log_file.close()
	val_log_file.close()
