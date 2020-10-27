###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import time
import cv2
import torch
import torchvision.transforms as transform
import numpy as np
import encoding.utils as utils

from tqdm import tqdm

from torch.utils import data
# from scipy.stats import multivariate_normal
from torch.distributions import MultivariateNormal

import libmr

from encoding.nn import BatchNorm
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_segmentation_model, MultiEvalModule


from option import Options
from classlabel import Category
def get_class_lists():
	data = open("logs/resnet.txt",'r').readlines()
	class_info = []
	for i in data:
		ratio,classes = i.split('|')
		ratio = float(ratio.split(':')[1])
		classes = classes.strip().split(':')[1].split(',')
		class_info.append((ratio,classes))
	return class_info

def build_gaussian(mean_weights,var_weights):
	gaussians = {}
	for key,val in mean_weights.items():
		category = int(key.split('_')[1])
		var_cat = "cov_{}".format(category)
		small_diag = np.ones(var_weights[var_cat].shape) * 0.01
		var = torch.tensor(var_weights[var_cat] + np.diag(np.diag(small_diag)))
		var = var.type(torch.cuda.FloatTensor)
		gaussians[category] = MultivariateNormal(val.cuda(),var)
	return gaussians

def build_weibull(features,ng=10):
	'''
	features: num_correct_samples x k
	'''
	weibulls = {}

	pred_class = torch.argmax(features,dim=1)
	feature_means = torch.zeros((features.size(1),1)).cuda()
	feature_sum = features.gather(1,pred_class.unsqueeze(dim=1))

	for i in range(features.size(1)):
		points = (pred_class == i).nonzero()
		feature_means[i] = torch.sum(feature_sum[points]) / points.size(0)
		weibull = libmr.MR()
		weibull.fit_high(torch.abs(feature_sum[points][:,0,0] - feature_means[i]),ng)
		weibulls[i+1] = weibull
	return weibulls,feature_means

def thresholding(weibulls,feature_means,test_data,objectness,eps):
	
	target_class = torch.argmax(test_data,dim=1)
	#print (target_class.size())
	#print (test_data.size())
	objects = torch.nonzero(objectness,as_tuple=True)
	#print (objects)
	test_data = test_data[objects[0],:,objects[1],objects[2]]
	#print (test_data.size())
	dist = torch.abs(feature_means.squeeze() - test_data)
	#target_class = torch.argmax(test_data,dim=1)[objects[0],:,objects[1],objects[2]]
	target_class = target_class[objects]
	#print (target_class.size())
	#print (dist.size())
	#print (weibulls.keys())
	#print (target_class[:100])
	#print (dist[:100,:].size())
	dist = dist[:,:].gather(0,target_class[:].unsqueeze(dim=1))
	#weibull_cdf = torch.Tensor([weibulls[i+1].cdf(dist[i]) for i in range(feature_means.size(1))])
	#print (weibulls[7],dist[6])
	#print (dist.size())
	#print ((target_class[0]+1).item())
	#print (weibulls[(target_class[0]+1).item()])
	#print (dist[0])
	print (dist)
	weibull_cdf = torch.Tensor([weibulls[(target_class[k]+1).item()].cdf(dist[k]) for k in range(dist.size(0))])
	outliers = (weibull_cdf > eps).nonzero()	
	return weibull_cdf,outliers


def test(args,classes):
	# output folder
	outdir = os.path.join(args.save_folder,str(int(args.ratio*10)))
	# outdir = "../results"
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	
	edge_outdir = os.path.join(outdir,'edge')
	if not os.path.exists(edge_outdir):
		os.makedirs(edge_outdir)

	objectness_outdir = os.path.join(outdir,'objectness')
	if not os.path.exists(objectness_outdir):
		os.makedirs(objectness_outdir)

	mask_outdir = os.path.join(outdir,'mask')
	if not os.path.exists(mask_outdir):
		os.makedirs(mask_outdir)
	# data transforms
	input_transform = transform.Compose([
		transform.ToTensor(),
		transform.Normalize([.485, .456, .406], [.229, .224, .225])])
	label_transform = transform.ToTensor()
	# dataset
	data_kwargs = {'transform': input_transform, 'target_transform':input_transform,
						'label_transform':label_transform}
	testset = get_segmentation_dataset(args.dataset,args.ratio,args.size, split=args.split, mode='test',
										   **data_kwargs)
	# dataloader
	loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
		if args.cuda else {}
	test_data = data.DataLoader(testset, batch_size=8,shuffle=False, **loader_kwargs)
	# model
	nclass = len(classes)
	if args.model_zoo is not None:
		model = get_model(args.model_zoo, pretrained=True)
	else:
		model = get_segmentation_model(args.ratio,nclass,args.model, dataset = args.dataset,
									   backbone = args.backbone, dilated = args.dilated,
									   lateral = args.lateral, jpu = args.jpu, aux = args.aux,
									   se_loss = args.se_loss, norm_layer = BatchNorm,
									   base_size = args.base_size, crop_size = args.crop_size)
		# resuming checkpoint
		if args.resume is None or not os.path.isfile(args.resume):
			raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
		checkpoint = torch.load(args.resume)
		# strict=False, so that it is compatible with old pytorch saved models
		model.load_state_dict(checkpoint['state_dict'])
		print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
	print (args.test_batch_size)
	#print(model)
	scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
		[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
	if not args.ms:
		scales = [1.0]
	#evaluator = MultiEvalModule(model, testset.num_class, scales=scales, flip=args.ms).cuda()
	evaluator = model.cuda()
	evaluator.eval()
	metric = utils.SegmentationMetric(nclass)

	tbar = tqdm(test_data)
	ids = testset._load_image_set_index()
	test_log = open("logs/{}.txt".format(int(args.ratio*10)),'w')
	overallpix = 0.0
	overallmIoU = 0.0
	#load gaussian model
	#mean_weights = torch.load("../models/gaussian/mean_{}.pt".format(int(args.ratio*10)))
	#var_weights = torch.load("../models/gaussian/var_{}.pt".format(int(args.ratio*10)))
	#gaussians = build_gaussian(mean_weights,var_weights)
	feature_data = torch.load("../models/weibull/{}.pt".format(int(args.ratio*10)))
	weibulls,feature_means = build_weibull(feature_data)
	category = Category(classes,True)
	threshold = 0.1
	for i, (image,labels,objectness,edge) in enumerate(tbar):
		print (i,image.size())
		image = image.type(torch.cuda.FloatTensor)
		# pass
		if 'val' in args.mode:
			with torch.no_grad():
				predicts = evaluator.parallel_forward(image)
				predicts = torch.argmax(predicts[0],dim=1)
				metric.update(labels[0], predicts)

				pixAcc, mIoU = metric.get()
				overallpix += pixAcc
				overallmIoU += mIoU
				tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
		else:
			with torch.no_grad():
				tic = time.time()
				outputs,objectness,edge_label = evaluator.val_forward(image)
				
				predict = torch.argmax(outputs,1)+2 #batch_size x 1 x H x W
				print (torch.unique(predict))
				objectness_pred = torch.argmax(objectness,dim=1) #batch_size x 1 x H x W
				predict = predict * objectness_pred
				print (torch.unique(predict))
				edge_pred = torch.argmax(edge_label,dim=1)
				#thresholding here
				weibull_cdfs,outliers = thresholding(weibulls,feature_means,outputs,objectness_pred,threshold)
				print (weibull_cdfs)
				print (outliers)
				print (torch.unique(weibull_cdfs))
				predict = predict * (1-edge_pred)
				toc = time.time()
				mask = utils.get_mask_pallete(predict, args.dataset)
				labels = labels.squeeze().cuda()
				pixAcc,mIoU,correct_classified = utils.batch_pix_accuracy(predict.data, labels)
				#thresholding(gaussians,category,threshold,features,correct_classified,predict)
				test_log.write('pixAcc:{:.4f},mIoU:{:.4f},cost:{:.3f}s\n'.format(pixAcc, mIoU,toc-tic))
				
				#record the accuracy
				metric.update(labels, predict.data)
				pixAcc, mIoU = metric.get()
				overallpix += pixAcc
				overallmIoU += mIoU
				#write the output
				#print (image[0].data.cpu().numpy())
				#cv2.imwrite(os.path.join("../experiments/results/truth0",outname),image[0].data.cpu().numpy().transpose(1,2,0))
				for j in range(8):
					outname = str(ids[i*8+j]) + '.png'
					cv2.imwrite(os.path.join(mask_outdir,outname),mask[j])

					objectness_output = objectness_pred[j].squeeze().cpu().numpy() * 255
					cv2.imwrite(os.path.join(objectness_outdir,outname),objectness_output)

					edge_output = edge_pred[j].squeeze().cpu().numpy()*255
					cv2.imwrite(os.path.join(edge_outdir, outname),edge_output)
		print ("Overall pixel accuracy:{:.4f},Overall mIoU:{:.4f}".format(pixAcc,mIoU))
	test_log.close()


if __name__ == "__main__":
	args = Options().parse()
	torch.manual_seed(args.seed)
	args.test_batch_size = torch.cuda.device_count()
	class_info = get_class_lists()

	test(args,class_info[0][1])
