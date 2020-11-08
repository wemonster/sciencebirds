###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import numpy as np
from tqdm import tqdm
import copy,math
import cv2
import random
from PIL import Image


import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.nn.customize import FocalLoss
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model
from encoding.roi_data_layer.roibatchLoader import roibatchLoader
from encoding.roi_data_layer.roidb import combined_roidb

from option import Options
# from maskimage import generate_dataset
from classlabel import Category
torch_ver = torch.__version__[:3]

from torch.autograd import Variable

class Trainer():
	def __init__(self,info,id_info,args):
		self.filename,self.classes = info[0],info[1]
		self.categories = id_info
		self.args = args
		# data transforms
		input_transform = transform.Compose([
			transform.ToTensor(),
			transform.Normalize([.485, .456, .406], [.229, .224, .225])
			])
		label_transform = transform.ToTensor()
		self.nclass = len(self.classes)
		#initialise the tensor holder here
		self.im_info = torch.FloatTensor(1)
		self.num_boxes = torch.LongTensor(1)
		self.gt_boxes = torch.FloatTensor(1)

		use_cuda = True
		if use_cuda:
			self.im_info = self.im_info.cuda()
			self.num_boxes =self.num_boxes.cuda()
			self.gt_boxes = self.gt_boxes.cuda()
		self.im_info = Variable(self.im_info)
		self.num_boxes = Variable(self.num_boxes)
		self.gt_boxes = Variable(self.gt_boxes)
		# dataset
		data_kwargs = {'transform': input_transform, 'target_transform':input_transform,
						'label_transform':label_transform}
		#get roidb info
		#train_imdb,train_roidb = combined_roidb('train',self.categories)
		#val_imdb,val_roidb = combined_roidb('val',self.categories)

		#get image info
		trainset = get_segmentation_dataset(args.dataset,self.filename,args.size, split=args.train_split, mode='train',
										   **data_kwargs)

		# trainLoader = roibatchLoader(train_roidb,trainset,self.nclass,mode='train')

		valset = get_segmentation_dataset(args.dataset,self.filename,args.size, split='val', mode ='val',
										   **data_kwargs)
		# valLoader = roibatchLoader(val_roidb,valset,self.nclass,mode='val')
		
		
		# dataloader
		kwargs = {'num_workers': args.workers, 'pin_memory': True} \
			if args.cuda else {}
		self.trainloader = data.DataLoader(trainset,batch_size=args.batch_size,
										   drop_last=True, shuffle=True, **kwargs)
		self.valloader = data.DataLoader(valset,batch_size=args.batch_size,
										   drop_last=False, shuffle=False, **kwargs)

		
		# trainloader = data.DataLoader(trainLoader, batch_size=args.batch_size,
		# 								   drop_last=True, shuffle=True, **kwargs)
		# valloader = data.DataLoader(valLoader, batch_size=args.batch_size,
		# 								 drop_last=False, shuffle=False, **kwargs)

		# self.dataloader = {'train':trainloader,'val':valloader}
		print ("finish loading the dataset")

		# # model
		model = get_segmentation_model(self.nclass,args.model, dataset = args.dataset,
									   backbone = args.backbone, dilated = args.dilated,
									   lateral = args.lateral, jpu = args.jpu, aux = args.aux,
									   se_loss = args.se_loss, #norm_layer = SyncBatchNorm,
									   base_size = args.base_size, crop_size = args.crop_size)


		# for (name,w) in model.named_parameters():
		# 	print (name,w.requires_grad)
		# optimizer using different LR
		params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr}]
		params_list.append({'params':model.low_level_1.parameters(),'lr':args.lr})
		params_list.append({'params':model.low_level_2.parameters(),'lr':args.lr})
		params_list.append({'params':model.concat_conv_1.parameters(),'lr':args.lr})
		params_list.append({'params':model.concat_conv_2.parameters(),'lr':args.lr})
		params_list.append({'params':model.objectness.parameters(),'lr':args.lr})
		params_list.append({'params':model.edge_conv.parameters(),'lr':args.lr})
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
		# self.criterion = FocalLoss(num_class = self.nclass,alpha=torch.ones((self.nclass,1))*0.25)
		#self.criterion = FocalLoss()
		self.model, self.optimizer = model, optimizer
		# using cuda
		if args.cuda:
			self.model = self.model.cuda()
			self.criterion = self.criterion.cuda()
		# 	self.model = DataParallelModel(self.model).cuda()
		# 	self.criterion = DataParallelCriterion(self.criterion).cuda()
		# resuming checkpoint
		self.best_pred = 0.0
		if args.resume is not None:
			if not os.path.isfile(args.resume):
				raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			if args.cuda:
				self.model.load_state_dict(checkpoint['state_dict'])
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
		self.correct_class = torch.tensor([])
	def training(self, epoch,log_file):
		train_loss = 0.0
		self.model.train()
		tbar = tqdm(self.trainloader)
		#data_iter = iter(self.dataloader['train'])
				
		for i, (image,labels,objectness,edge) in enumerate(tbar):
			#img_data = data_iter.next()
			#self.im_info.resize_(img_data[0].size()).copy_(img_data[0])
			#self.gt_boxes.resize_(img_data[1].size()).copy_(img_data[1])
			#self.num_boxes.resize_(img_data[2].size()).copy_(img_data[2])


			image = image.type(torch.cuda.FloatTensor)
			
			self.scheduler(self.optimizer, i, epoch, self.best_pred)
			self.optimizer.zero_grad()
			pixel_wise,cat_label,edge_label = self.model(image,self.im_info,self.gt_boxes,self.num_boxes)
			pixel_wise = pixel_wise.type(torch.cuda.FloatTensor)
			cat_label = cat_label.type(torch.cuda.FloatTensor)
			edge_label = edge_label.type(torch.cuda.FloatTensor)

			labels = torch.squeeze(labels)
			labels = labels.to(dtype=torch.int64).cuda()
			
			edge = torch.squeeze(edge)
			edge = edge.to(dtype=torch.int64).cuda()

			objectness = torch.squeeze(objectness)
			objectness = objectness.to(dtype=torch.int64).cuda()
			#only takes account those foregrounds
			foregrounds = (labels > 0).nonzero()
			batch,x,y = foregrounds[:,0],foregrounds[:,1],foregrounds[:,2]
#			print (labeled.size(),labels.size())
			cat_label = cat_label[batch,:,x,y]
			labels = labels[batch,x,y] - 2
#			print (labeled.size(),pixel_wise.size())
#			print (torch.unique(labels),torch.unique(objectness))
		#	print (cat_label.size(),labels.size())
		#	print ("-"*20)
			class_loss = self.criterion(cat_label, labels)
			objectness_loss = self.criterion(pixel_wise,objectness)
			edge_loss = self.criterion(edge_label,edge)
			loss = class_loss.item() + objectness_loss.item()
			loss = class_loss + objectness_loss + edge_loss
#			print (loss)
			loss.backward()
			self.optimizer.step()
			train_loss += loss.item()
			tbar.set_description('Train loss:{:.3f}'
				.format(train_loss / (i + 1)))
		log_file.write("Epoch:{}, Loss:{:.3f}\n".format(epoch,train_loss/(i+1)))
		if self.args.no_val:
			# save checkpoint every epoch
			is_best = False
			utils.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': self.model.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'best_pred': self.best_pred,
			}, self.args, is_best, self.filename,filename='checkpoint_{}.pth.tar'.format(epoch))


	def validation(self, epoch,log_file):
		# Fast test during the training
		def collect_features(features,position,pred):
			'''
			features: batch_size x k x H x W (k is number of known class)
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
				target = features[img,:,x,y]
				target_class = torch.argmax(target,dim=1)
				#print (target.size(),target_class.size())
				if len(self.correct_features) == 0:
					self.correct_features = target
				else:
					self.correct_features = torch.cat((self.correct_features,target))

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
				target = features[:,x,y]
				if len(self.correct_features) == 0:
					self.correct_features = target
				else:
					self.correct_features = torch.cat((self.correct_features,target))



		def eval_batch(epoch,model, image, target,object_truth,edge):
			labeled,objectness,edge_label = model.val_forward(image)
			objectness_pred = torch.argmax(objectness,dim=1) #batch_size x 1 x H x W
			object_truth = object_truth.squeeze().cuda()
			pred = torch.argmax(labeled,dim=1)+2 #batch_size x 1 x H x W
			pred = objectness_pred * pred
			target = target.squeeze().cuda()
		
			edge_pred = torch.argmax(edge_label,dim=1)
			edge = edge.squeeze().cuda()

			edge_correct,edge_labeled,edge_correct_classified = utils.batch_pix_accuracy(edge_pred.data,edge)
			correct, cat_labeled,correct_classified = utils.batch_pix_accuracy(pred.data, target)

			correct_object,labeled_object,correct_classified_object = utils.batch_pix_accuracy(objectness_pred.data,object_truth)
			if epoch > 5:
				collect_features(labeled,correct_classified,pred)
				self.build_weibull_model()
			inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
			return correct, cat_labeled, inter, union, correct_object, labeled_object,edge_correct,edge_labeled

		

		is_best = False
		self.model.eval()
		total_inter, total_union, total_correct, total_label, total_object,total_object_label,total_edge,total_edge_label = 0, 0, 0, 0, 0, 0, 0, 0
		tbar = tqdm(self.valloader, desc='\r')
		for i, (image,labels,objectness,edge) in enumerate(tbar):
			image = image.type(torch.cuda.FloatTensor)
			if torch_ver == "0.3":
				image = Variable(image, volatile=True)
				correct, labeled, inter, union, correct_object,labeled_object,edge_correct,edge_labeled = eval_batch(self.model, image, labels,objectness,edge)
			else:
				with torch.no_grad():
					correct, labeled, inter, union, correct_object,labeled_object,edge_correct,edge_labeled = eval_batch(epoch,self.model, image, labels,objectness,edge)

			total_correct += correct
			total_label += labeled
			total_inter += inter
			total_union += union
			total_object += correct_object
			total_object_label += labeled_object
			total_edge += edge_labeled
			total_edge_label += edge_labeled
			pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
			objAcc = 1.0 * total_object / (np.spacing(1) + total_object_label)
			edgAcc = 1.0 * total_edge / (np.spacing(1) + total_edge_label)
			IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
			mIoU = IoU.mean()
			tbar.set_description(
				'pixAcc: %.3f, mIoU: %.3f, objAcc: %.3f, edgAcc: %.3f' % (pixAcc, mIoU,objAcc,edgAcc))
		new_pred = (pixAcc + mIoU + objAcc + edgAcc)/4
		log_file.write("Epoch:{}, pixAcc:{:.3f}, mIoU:{:.3f}, Overall:{:.3f}\n".format(epoch,pixAcc,mIoU,new_pred))
		if new_pred >= self.best_pred:
			is_best = True
			self.best_pred = new_pred
			utils.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': self.model.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'best_pred': new_pred,
			}, self.args, is_best,self.filename,"checkpoint_{}.pth.tar".format(epoch+1))

	def build_gaussian_model(self):
		occurrance = self.corresponding_class.cpu().numpy()
		(category,occurrance) = np.unique(occurrance,return_counts=True)
		class_mean = {}
		class_var = {}
		if not os.path.exists("../models/weibull"):
			os.mkdir("../models/weibull")
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

		torch.save(class_mean,os.path.join("../models/weibull","{}.pt".format(self.filename)))

	def build_weibull_model(self):
		if not os.path.exists("../models/weibull"):
			os.mkdir("../models/weibull")
		torch.save(self.correct_features,os.path.join("../models/weibull","{}.pt".format(self.filename)))



def get_class_lists():
	data = open("logs/resnet.txt",'r').readlines()
	class_info = []
	for i in data:
		filename,classes = i.split('|')
		classes = classes.strip().split(',')
		class_info.append((filename,classes))
	return class_info


if __name__ == "__main__":
	args = Options().parse()
	torch.manual_seed(args.seed)
	class_info = get_class_lists()
	root = "logs/{}".format(args.size)
	if not os.path.exists(root):
		os.mkdir(root)
	for i in range(len(class_info)):
		id_info = Category(class_info[i][1])
		trainer = Trainer(class_info[i],id_info,args)
		filename = class_info[i][0]
		print (filename)
		train_log_file = open(os.path.join(root,"training_{}_log.txt".format(filename)),'w')
		val_log_file = open(os.path.join(root,"val_{}_log.txt".format(filename)),'w')
		print('Starting Epoch:', trainer.args.start_epoch)
		print('Total Epoches:', trainer.args.epochs)
		for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
		 	trainer.training(epoch,train_log_file)
		 	if not trainer.args.no_val:
		 		trainer.validation(epoch,val_log_file)
		trainer.build_weibull_model()
		train_log_file.close()
		val_log_file.close()
