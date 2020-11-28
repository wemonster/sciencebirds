###########################################################################
# Created by: Jianan Yang
# Email: u7083746@anu.edu.au
# Copyright (c) 2020
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
from test_accuracy import *
def get_class_lists():
	data = open("logs/resnet.txt",'r').readlines()
	class_info = {}
	for i in data:
		filename,classes = i.split('|')
		classes = classes.strip().split(',')
		class_info[filename] = classes
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

def build_weibull(features,ng=50):
	'''
	features: num_correct_samples x k
	'''
	weibulls_high = {}
	weibulls_low = {}
	pred_class = torch.argmax(features,dim=1)
	feature_means = torch.zeros((features.size(1),1)).cuda()
	feature_sum = features.gather(1,pred_class.unsqueeze(dim=1))

	

	for i in range(features.size(1)):
		points = (pred_class == i).nonzero()
		feature_means[i] = torch.sum(feature_sum[points]) / points.size(0)
		weibull_high = libmr.MR()
		weibull_low = libmr.MR()
		#print (points.size())
		toptail = torch.sort(torch.abs(feature_sum[points][:,0,0] - feature_means[i]))[0]
		# print (len(toptail))
		weibull_high.fit_high(toptail,len(toptail))
		bottail = torch.sort(torch.abs(feature_sum[points][:,0,0] - feature_means[i]),descending=False)[0]
		weibull_low.fit_low(bottail,len(bottail))
		weibulls_high[i+1] = weibull_high
		weibulls_low[i+1] = weibull_low
	return weibulls_high,weibulls_low,feature_means

def recalibrate_scores(weibull_model,feature_means,test_data,objectness,alpharank,num_classes,eps):

	target_class = torch.argmax(test_data,dim=1)
	#print (target_class.size())
	#print (test_data.size())
	objects = torch.nonzero(objectness,as_tuple=True)
	#print (objects)
	test_data = test_data[objects[0],:,objects[1],objects[2]] #N x k, n is number of samples and k is dim
	#print (test_data.size())
	test_data = torch.abs(feature_means.squeeze() - test_data)
	# print (test_data[0])
	# print (dist.size())
	#get top alpha index for each feature
	target_class = target_class[objects]
	# dist = dist[:,:].gather(0,target_class[:].unsqueeze(dim=1))
	# print (feature_means)
	ranked_vals = test_data.sort(dim=1,descending=True)[0]
	# print (ranked_vals)
	ranked_ids = test_data.argsort(dim=1,descending=True)
	# print(ranked_ids.shape)
	# print (ranked_ids[0])
	alpha_weights = [(1) / float(alpharank) for i in range(1,alpharank+1)]
	# print (alpha_weights)
	ranked_alpha = torch.zeros((ranked_ids.size(0),num_classes)).cuda()
	# ranked_alpha
	for i in range(len(alpha_weights)):
		# print (ranked_ids[:,i])
		ranked_alpha[range(ranked_ids.size(0)),ranked_ids[:,i]] = alpha_weights[i]
	# print (ranked_alpha[0])
	# tic = time.time()
	# scores = torch.Tensor([weibull_model[(ranked_ids.flatten()[i]+1).item()].w_score(ranked_vals.flatten()[i].item()) for i in range(len(ranked_ids.flatten()))]).cuda()
	scores = torch.Tensor([weibull_model[i%num_classes+1].w_score(test_data.flatten()[i]) for i in range(len(ranked_ids.flatten()))]).cuda()
	# toc = time.time()
	# print (toc-tic)

	# print (scores)
	# print (max(scores),min(scores))
	# # res = func(ranked_vals.flatten(),ranked_ids.flatten())
	# modified_scores = torch.Tensor([weibull_model[k+1].w_score(ranked_vals)])
	w_scores = ranked_alpha * scores.reshape(ranked_ids.shape)
	# print (w_scores[0])
	# w_scores[w_scores==0] = 1
	revised = test_data * w_scores
	# print ("revised",revised[0])
	# tic = time.time()
	# # modified_scores = test_data - revised
	modified_score = revised.sum(axis=1)
	expanded = torch.empty((ranked_ids.size(0),num_classes+1)).cuda()
	# toc = time.time()
	# print (toc - tic)
	expanded[:,0] = modified_score
	w_scores[w_scores==0] = 1
	expanded[:,1:] = test_data * w_scores
	# print (test_data[:5])
	# print (expanded[:5])
	# # print (torch.nn.functional.softmax(test_data,dim=1)[:5])
	# # print (torch.nn.functional.softmax(expanded,dim=1)[:5])
	recalibrated_score = torch.nn.Softmax(dim=1)(expanded)
	# pred = torch.argmax(expanded,dim=1)
	print (recalibrated_score.max(dim=1)[0][:10])
	outliers = (recalibrated_score.max(dim=1)[0]<eps).nonzero().squeeze()
	print (len(outliers),test_data.size(0))
	# pred = recalibrated_score.argmax(dim=1)+2
	# pred[torch.max(recalibrated_score,dim=1)[0] < eps] = 1
	return objects,outliers
	# openmax_unknown = []
	# for k in range(dist.size(0)):
	# 	print (dist[k])
	# for i in range(num_classes):
		# wscore = [weibull_model[i+1].w_score(dist[k]) for k in range(dist.size(0))]

def threshold_on_softmax(objectness,test_data,eps):
	objects = torch.nonzero(objectness,as_tuple=True)
	test_data = test_data[objects[0],:,objects[1],objects[2]]
	# print (test_data[:5])
	probs = torch.nn.Softmax(dim=1)(test_data) # b x k x h x w
	# print (probs[:5])
	print (probs.max(dim=1)[0][:10])
	outliers = (probs.max(dim=1)[0] < eps).nonzero().squeeze()
	# print (len(outliers))
	return objects,outliers


def thresholding(weibulls_high,weibulls_low,feature_means,test_data,objectness,eps):
	
	target_class = torch.argmax(test_data,dim=1)
	#print (target_class.size())
	#print (test_data.size())
	objects = torch.nonzero(objectness,as_tuple=True)
	#print (objects)
	test_data = test_data[objects[0],:,objects[1],objects[2]]
	print (test_data.size())
	#print (test_data.size())
	print (feature_means)
	dist = torch.abs(feature_means.squeeze() - test_data)
	#target_class = torch.argmax(test_data,dim=1)[objects[0],:,objects[1],objects[2]]
	target_class = target_class[objects]
	dist = dist[:,:].gather(0,target_class[:].unsqueeze(dim=1))

	print (dist)
	weibull_high_cdf = torch.Tensor([weibulls_high[(target_class[k]+1).item()].cdf(dist[k]) for k in range(dist.size(0))])
	weibull_low_cdf = torch.Tensor([weibulls_low[(target_class[k]+1).item()].cdf(dist[k]) for k in range(dist.size(0))])
	# print (torch.max(weibull_high_cdf),torch.min(weibull_high_cdf))
	# print (torch.max(weibull_low_cdf),torch.min(weibull_low_cdf))
	high_outliers = (weibull_high_cdf > 1-eps).nonzero().squeeze()	
	low_outliers = (weibull_low_cdf > eps).nonzero().squeeze()
	
	print (objects[0].size(),high_outliers.size(),low_outliers.size())
	#print (objects[0][outliers].size(),objects[1][outliers].size(),objects[2][outliers].size())

	return objects,high_outliers,low_outliers


def test(args,model_name,classes,threshold):
	# output folder
	outdir = os.path.join(args.save_folder,model_name)
	# outdir = "../results"
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	
	edge_outdir = os.path.join(outdir,'edge')
	if not os.path.exists(edge_outdir):
		os.makedirs(edge_outdir)

	objectness_pred_outdir = os.path.join(outdir,"objectness_pred")
	if not os.path.exists(objectness_pred_outdir):
		os.makedirs(objectness_pred_outdir)

	objectness_outdir = os.path.join(outdir,'objectness')
	if not os.path.exists(objectness_outdir):
		os.makedirs(objectness_outdir)

	refinement_outdir = os.path.join(outdir,"refine")
	if not os.path.exists(refinement_outdir):
		os.makedirs(refinement_outdir)

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
	testset = get_segmentation_dataset(args.dataset,model_name,args.size, split=args.split, mode='test',
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
		model = get_segmentation_model(nclass,args.model,model_name, dataset = args.dataset,
									   backbone = args.backbone, dilated = args.dilated,
									   lateral = args.lateral, jpu = args.jpu, aux = args.aux,
									   se_loss = args.se_loss, norm_layer = BatchNorm,
									   base_size = args.base_size, crop_size = args.crop_size)
		# resuming checkpoint
		#if args.resume is None or not os.path.isfile(args.resume):
		#	raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
		#checkpoint = torch.load(args.resume)
		checkpoint = torch.load("../experiments/runs/characters/model_best_{}.pth.tar".format(model_name))
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
	test_log = open("logs/{}.txt".format(model_name),'w')
	overallpix = 0.0
	overallmIoU = 0.0
	#load gaussian model
	#mean_weights = torch.load("../models/gaussian/mean_{}.pt".format(int(args.ratio*10)))
	#var_weights = torch.load("../models/gaussian/var_{}.pt".format(int(args.ratio*10)))
	#gaussians = build_gaussian(mean_weights,var_weights)
	feature_data = torch.load("../models/weibull/{}.pt".format(model_name))
	weibulls_high,weibulls_low,feature_means = build_weibull(feature_data)
	category = Category(classes,True)
	alpharank = nclass // 2
	for i, (image,labels,objectness_truth,edge) in enumerate(tbar):
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
				#print (torch.unique(predict))
				objectness_pred = torch.argmax(objectness,dim=1) #batch_size x 1 x H x W
				# predict = predict * objectness_pred
				#print (torch.unique(predict))
				edge_pred = torch.argmax(edge_label,dim=1)
				#thresholding here
				# objects,outliers = threshold_on_softmax(objectness_pred,outputs,threshold)
				objects,outliers = recalibrate_scores(weibulls_high,feature_means,outputs,objectness_pred,alpharank,nclass,threshold)
				# objects,high_outliers,low_outliers = thresholding(weibulls_high,weibulls_low,feature_means,outputs,objectness_pred,threshold)
				objectness_truth = objectness_truth.type(torch.cuda.LongTensor).squeeze(dim=1)
				objectness_truth[objectness_truth>0] = 1
				# print (objectness_pred==0)
				# predict[objectness_pred==0] = 1
				# predict[predict.nonzero()] = 1
				# print (predict.size(),objectness_truth.size())
				
				#print (weibull_cdfs)
				#print (outliers)
				#print (outliers.size())
				#print (high_outliers.size(),low_outliers.size())
				#print (torch.unique(weibull_cdfs))
				predict[objects[0][outliers],objects[1][outliers],objects[2][outliers]] = 1
				#predict[objects[0][high_outliers],objects[1][high_outliers],objects[2][high_outliers]] = 1
				#predict[objects[0][low_outliers],objects[1][low_outliers],objects[2][low_outliers]] = 1
				predict = predict * objectness_truth
				# predict = predict * (1-edge_pred)
				# toc = time.time()
				# #mask = utils.get_mask_pallete(predict, category,args.dataset)
				# labels = labels.squeeze().cuda()
				# pixAcc,mIoU,correct_classified = utils.batch_pix_accuracy(predict.data, labels)
				# #thresholding(gaussians,category,threshold,features,correct_classified,predict)
				# test_log.write('pixAcc:{:.4f},mIoU:{:.4f},cost:{:.3f}s\n'.format(pixAcc, mIoU,toc-tic))
				
				# #record the accuracy
				# metric.update(labels, predict.data)
				# pixAcc, mIoU = metric.get()
				# overallpix += pixAcc
				# overallmIoU += mIoU
				#write the output
				#print (image[0].data.cpu().numpy())
				#cv2.imwrite(os.path.join("../experiments/results/truth0",outname),image[0].data.cpu().numpy().transpose(1,2,0))
				for j in range(8):
					outname = str(ids[i*8+j]) + '.png'
					# print (torch.unique(predict[j]))

					cv2.imwrite(os.path.join(mask_outdir,outname),predict[j].squeeze().cpu().numpy())

					objectness_output = objectness_truth[j].squeeze().cpu().numpy() * 255
					cv2.imwrite(os.path.join(objectness_outdir,outname),objectness_output)

					objectness_pred_output = objectness_pred[j].squeeze().cpu().numpy() * 255
					cv2.imwrite(os.path.join(objectness_pred_outdir,outname),objectness_pred_output)
					
					objectness_finement = predict * (1-edge_pred)
					refine_output = objectness_finement[j].squeeze().cpu().numpy()
					cv2.imwrite(os.path.join(refinement_outdir,outname),refine_output)

					edge_output = edge_pred[j].squeeze().cpu().numpy()*255
					cv2.imwrite(os.path.join(edge_outdir, outname),edge_output)
		# print ("Overall pixel accuracy:{:.4f},Overall mIoU:{:.4f}".format(pixAcc,mIoU))
	test_log.close()


if __name__ == "__main__":
	args = Options().parse()
	torch.manual_seed(args.seed)
	args.test_batch_size = torch.cuda.device_count()
	class_info = get_class_lists()
	result_file = open("openmax.txt",'w')
	#print (class_info)
	# for model_name in class_info.keys():
	# 	if model_name.startswith("00"):
	for model_name in ['00','01','02','03','04','05','BLACKBIRD','HILL','REDBIRD','WOOD','YELLOWBIRD','ICE']:
		#	continue
		print (model_name)
		result_file.write("{}\t".format(model_name))
		for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
			test(args,model_name,class_info[model_name],threshold)
			groundtruth_folder = "../dataset/rawdata/groundtruth"
			ratio_truth_folder = "../dataset/images/small"
			test_image_folder = "../experiments/results"

			testsets = os.listdir(test_image_folder)

			# for filename in testsets:
			filename = model_name
			truthroot = os.path.join(ratio_truth_folder,filename)
			resultroot = os.path.join(test_image_folder,filename)
			#edge
			# edge_accuracy(truthroot,resultroot)
			#objectness
			pixAcc,mIoU,refined_pixAcc,refined_mIoU = mask_accuracy(truthroot,resultroot,len(class_info[model_name]))
			result_file.write("{:.4f}|{:.4f}|{:.4f}|{:.4f}\t".format(pixAcc,mIoU,refined_pixAcc,refined_mIoU))
		result_file.write("\n")