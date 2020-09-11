from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import copy
import time,math
import numpy as np
import random
import encoding.dilated as resnet
import matplotlib.pyplot as plt

HILL = 'hill'
SLING = 'slingshot'
REDBIRD = 'redBird'
YELLOWBIRD = 'yellowBird'
BLUEBIRD = 'blueBird'
BLACKBIRD = 'blackBird'
WHITEBIRD = 'whiteBird'
PIG = 'pig'
ICE = 'ice'
WOOD = 'wood'
STONE = 'stone'
TNT = 'TNT'

folder = "dataset/characters"

num_classes = 14
input_size = 224
batch_size = 4

num_epochs = 50

feature_extract = True

def train_model(model,dataloaders,criterion,optimizer,output_model,num_epochs=25,is_inception=False):
	since = time.time()
	val_acc_history = []
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	train_loss = []
	train_acc = []
	val_loss = []
	val_acc = []
	logs = open('logs/log.txt','w')
	for epoch in range(num_epochs):
		#print (model.state_dict()['layer5.1.bn1.weight'])
		for phase in ['train','val']:
			phase_tic = time.time()
			print ("Epoch {}/{}".format(epoch,num_epochs - 1))
			print ('-' * 10)

			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			for inputs,labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase=='train'):
					if is_inception and phase=='train':
						outputs,aux_outputs = model(inputs)
						loss1 = criterion(outputs,labels)
						loss2 = criterion(aux_outputs,labels)
						loss = loss1 + loss2
					else:
						outputs = model(inputs)
						loss = criterion(outputs,labels)
					_,preds = torch.max(outputs,1)
					
					if phase == 'train':
						#print (outputs.requires_grad)
						loss.backward()
						optimizer.step()
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
			phase_toc = time.time()
			logs.write("{} Loss: {:.4f} Acc: {}/{},{:.4f} cost: {:.4f}s\n\n".format(phase,epoch_loss,running_corrects,
				len(dataloaders[phase].dataset),epoch_acc,phase_toc-phase_tic))
			print ("{} Loss: {:.4f} Acc: {}/{},{:.4f} cost: {:.4f}s".format(phase,epoch_loss,running_corrects,
				len(dataloaders[phase].dataset),epoch_acc,phase_toc-phase_tic))
			eval("{}_loss.append(epoch_loss)".format(phase))
			eval("{}_acc.append(epoch_acc)".format(phase))

			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
			if phase == 'val':
				val_acc_history.append(epoch_acc)
	logs.close()
	time_elapsed = time.time() - since
	print ("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60,time_elapsed % 60))
	model.load_state_dict(best_model_wts)
	torch.save(best_model_wts,"../models/resnet/pretrained_{}.pkl".format(output_model))
	info = {'training_loss':train_loss,
			'training_acc':train_acc,
			'val_loss':val_loss,
			'val_acc':val_acc}
	return model,info



def set_parameter_requires_grad(model,feature_extract):
	if feature_extract:
		for param in model.parameters():
			param.requires_grad = False


def load_weights(pklfile):
	if torch.cuda.is_available():
		wts = torch.load('mynet.pkl')
	else:
		wts = torch.load('mynet.pkl',map_location=torch.device('cpu'))
	return wts
#initialise and reshape network
def initialise_model(num_classes,feature_extract,use_pretrained=True):
	# model_ft = models.resnet50(pretrained=use_pretrained)
	# set_parameter_requires_grad(model_ft,feature_extract)
	model = resnet.resnet50()

	# wts = model_ft.state_dict()
	# model_dict = model.state_dict()
	# # print(model_dict)
	# pretrained_dict = {k:v for k,v in wts.items() if k in model_dict and not k.startswith('fc') and not k.startswith('layer4')}
	# model_dict.update(pretrained_dict)
	# # print (model_dict)
	# #model.load_state_dict(model_dict)
	# set_parameter_requires_grad(model,feature_extract)
	# for name,param in model.named_parameters():
	# 	if name.startswith('layer4') or name.startswith('layer5') or name.startswith('fc'):
	# 		param.requires_grad = True
	# 	print (name,param.requires_grad)
	return model

def initialise_resnet_model(num_classes):
	#fine tuning the fully connected layer
	model = models.resnet50(pretrained=True)
	set_parameter_requires_grad(model,True)
	num_features = model.fc.in_features
	model.fc = nn.Linear(in_features,num_classes,bias=True)


def load_pretrained_weights(model,state_dict):
	model.load_state_dict(state_dict)
	return model

def load_data(folder,batch_size,ratio):
	data_transforms = {
	'train':transforms.Compose([

		transforms.Resize((input_size,input_size)),
		transforms.ToTensor(),
		#mean and var in Imagenet
		transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
		]),
	'val':transforms.Compose([
		transforms.Resize((input_size,input_size)),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
		])
	}
	files = os.listdir(folder)
	classes = os.listdir(os.path.join(folder,"train"))
	number_knowns = math.floor(len(classes) * (1-ratio))
	files = random.sample(classes,number_knowns)
	# imgs = [cv2.imread(os.path.join(folder,i)) for i in files]
	imgs = {x:datasets.ImageFolder(os.path.join(folder,x),transform=data_transforms[x]) 
	for x in ['train','val']}
	dataloaders_dict = {x:torch.utils.data.DataLoader(imgs[x],batch_size=batch_size,shuffle=True) for x in ['train','val']}
	# imgs = [data_transforms['train'](Image.open(os.path.join(folder,i))) for i in files]
	# image_datasets = {'train':data_transforms['train']}
	# for inputs,label in dataloaders_dict:
	# 	print (inputs,label)
	return dataloaders_dict,data_transforms,files


def optimizer(model):
	params_to_update = model.parameters()
	print ("Params to learn:")
	if feature_extract:
		params_to_update = []
		for name,param in model.named_parameters():
			if param.requires_grad == True:
				params_to_update.append(param)
				print ('\t',name)
	else:
		for name,param in model.named_parameters():
			if param.requires_grad == True:
				print ('\t',name)
	optimizer_model = optim.SGD(params_to_update,lr=0.001,momentum=0.9)
	return optimizer_model


def plot_info(titles,logs):
	logs = open(logs,'r')
	training_loss = []
	training_accuracy = []
	val_loss = []
	val_accuracy = []
	data = logs.readlines()
	for i in range(len(data)):
		print (i,data[i])
		if data[i].startswith('train'):
			#training info
			info = data[i].split(',')
			print (info)
			loss = info[0][12:18]
			accuracy = info[1][:6]
			training_loss.append(float(loss))
			training_accuracy.append(float(accuracy))
		elif data[i].startswith('val'):
			#val info
			info = data[i].split(',')
			loss = info[0][10:16]
			accuracy = info[1][:6]
			val_loss.append(float(loss))
			val_accuracy.append(float(accuracy))
	print (training_loss)
	print (training_accuracy)
	print (val_loss)
	print (val_accuracy)
	plt.figure()
	plt.plot(training_loss,label='training loss')
	plt.plot(val_loss,label="val loss")
	plt.title('{} loss'.format(titles))
	plt.legend(('training loss','val loss'))
	plt.show()

	plt.figure()
	plt.plot(training_accuracy,label="training accuracy")
	plt.plot(val_accuracy,label="val accuracy")
	plt.title('{} accuracy'.format(titles))
	plt.legend(('training accuracy',"val accuracy"))
	plt.show()


def test_model(model,weights):
	wts = torch.load(weights,map_location=torch.device('cuda:0'))
	imgs,data_transforms = load_data(folder,batch_size)
	model.load_state_dict(wts)
	model.eval()
	test_cases = 20
	classes = ['BLACKBIRD','BLUEBIRD','HILL','ICE','PIG','REDBIRD','ROUNDWOOD','SLING','STONE','TERRAIN','TNT','WHITEBIRD','WOOD','YELLOWBIRD']
	corrects = 0
	for i in range(test_cases):
		random_class = random.choice(classes)
		target_class = folder + '/train/{}'.format(random_class)
		files = os.listdir(target_class)
		test_file = random.choice(files)
		test_file = os.path.join(target_class,test_file)
		
		test_img = Image.fromarray(cv2.imread(test_file))
		tens = Variable(data_transforms['train'](test_img))
		# print (tens.size())
		tens = tens.view(1,3,224,224)
		preds = nn.Softmax()(model(tens)).data.cpu().numpy()
		print (preds)
		res = np.argmax(preds)
		corrects += res == classes.index(random_class)
		print (res)
		print (random_class,classes[res])
	print ("{}/{}".format(corrects,test_cases))


if __name__ == "__main__":

	close_ratios = [0,0.1,0.3,0.5,0.7,0.9]
	if not os.path.exists("../models/resnet"):
		os.mkdir("../models/resnet")
	resnet_log = open("logs/resnet.txt",'w')
	for i in range(len(close_ratios)):
		model = initialise_model(num_classes,feature_extract)
		dataloaders_dict,data_transforms,classes = load_data('dataset/characters',8,close_ratios[i])
		resnet_log.write("ratio:{}|known:{}\n".format(close_ratios[i],','.join(classes)))
		criterion = nn.CrossEntropyLoss()
		optimizer_model = optimizer(model)
		output_model = str(int(close_ratios[i] * 10))
		train_model(model,dataloaders_dict,criterion,optimizer_model,output_model,num_epochs)
	resnet_log.close()
	# test_model(model,"mynet.pkl")
	# plot_info("batchnorm",'logs/log.txt')
	# plot_info('instance norm','logs/instance_norm_log.txt')
	# for key,val in info.items():
		# plot_info(val,key)

	