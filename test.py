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

import encoding.utils as utils

from tqdm import tqdm

from torch.utils import data

from encoding.nn import BatchNorm
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from option import Options

def get_class_lists():
	data = open("logs/resnet.txt",'r').readlines()
	class_info = []
	for i in data:
		ratio,classes = i.split('|')
		ratio = float(ratio.split(':')[1])
		classes = classes.strip().split(':')[1].split(',')
		class_info.append((ratio,classes))
	return class_info
def test(args,classes):
	# output folder
	outdir = os.path.join(args.save_folder,str(int(args.ratio*10)))
	# outdir = "../results"
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	# data transforms
	input_transform = transform.Compose([
		transform.ToTensor(),
		transform.Normalize([.485, .456, .406], [.229, .224, .225])])
	label_transform = transform.ToTensor()
	# dataset
	data_kwargs = {'transform': input_transform, 'target_transform':input_transform,
						'label_transform':label_transform}
	testset = get_segmentation_dataset(args.dataset, split=args.split, mode='val',
										   **data_kwargs)
	# dataloader
	loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
		if args.cuda else {}
	test_data = data.DataLoader(testset, batch_size=args.test_batch_size,shuffle=False, **loader_kwargs)
	# model
	nclass = len(classes) + 1
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

	print(model)
	scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
		[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
	if not args.ms:
		scales = [1.0]
	#evaluator = MultiEvalModule(model, testset.num_class, scales=scales, flip=args.ms).cuda()
	evaluator = model
	evaluator.eval()
	metric = utils.SegmentationMetric(testset.num_class)

	tbar = tqdm(test_data)
	ids = testset._load_image_set_index()
	test_log = open("logs/{}.txt".format(args.experiment),'w')
	overallpix = 0.0
	overallmIoU = 0.0
	#load gaussian model
	mean_weights = torch.load("../models/gaussian/mean_{}.pt".format(int(args.ratio*10)))
	var_weights = torch.load("../models/gaussian/var_{}.pt".format(int(args.ratio*10)))

	for i, (image,labels) in enumerate(tbar):
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
				print (image.size())
				outputs,_ = evaluator(image)
				predict = torch.argmax(outputs,1)
				print (predict.size())
				#thresholding here
				toc = time.time()
				mask = utils.get_mask_pallete(predict, args.dataset)

				#record the accuracy
				metric.update(labels, predict.data)
				pixAcc, mIoU = metric.get()
				overallpix += pixAcc
				overallmIoU += mIoU
				test_log.write('pixAcc:{:.4f},mIoU:{:.4f},cost:{:.3f}s\n'.format(pixAcc, mIoU,toc-tic))
				#write the output
				outname = str(ids[i]) + '.png'
				cv2.imwrite(os.path.join(outdir, outname),mask)
		print ("Overall pixel accuracy:{:.4f},Overall mIoU:{:.4f}".format(overallpix/(i+1),mIoU/(i+1)))
		test_log.close()


if __name__ == "__main__":
	args = Options().parse()
	torch.manual_seed(args.seed)
	args.test_batch_size = torch.cuda.device_count()
	class_info = get_class_lists()

	test(args,class_info[0][1])
