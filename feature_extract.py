import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import encoding.dilated as resnet
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from encoding.models import get_segmentation_model
from option import Options



savepath='vis_ImageNet_resnet50/'
if not os.path.exists(savepath):
	os.mkdir(savepath)
 
 
def draw_features(width,height,x,savename):
	tic=time.time()
	fig = plt.figure(figsize=(16, 16))
	fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
	for i in range(width*height):
		plt.subplot(height,width, i + 1)
		plt.axis('off')
		# plt.tight_layout()
		img = x[0, i, :, :]
		pmin = np.min(img)
		pmax = np.max(img)
		img = (img - pmin) / (pmax - pmin + 0.000001)
		plt.imshow(img, cmap='gray')
		print("{}/{}".format(i,width*height))
	fig.savefig(savename, dpi=100)
	fig.clf()
	plt.close()
	print("time:{}".format(time.time()-tic))
 
 
class ft_net(nn.Module):

	def __init__(self):
		super(ft_net, self).__init__()
		model_ft = resnet.resnet50(pretrained=False)
		self.model = model_ft
 
	def forward(self, x):
		if True: # draw features or not
			x = self.model.conv1(x)
			draw_features(8,8,x.cpu().numpy(),"{}/f1_conv1.png".format(savepath))
 
			x = self.model.bn1(x)
			draw_features(8, 8, x.cpu().numpy(),"{}/f2_bn1.png".format(savepath))
 
			x = self.model.relu(x)
			draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))
 
			x = self.model.maxpool(x)
			draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))
 
			x = self.model.layer1(x)
			draw_features(16, 16, x.cpu().numpy(), "{}/f5_layer1.png".format(savepath))
 
			x = self.model.layer2(x)
			draw_features(16, 32, x.cpu().numpy(), "{}/f6_layer2.png".format(savepath))
 
			x = self.model.layer3(x)
			draw_features(32, 32, x.cpu().numpy(), "{}/f7_layer3.png".format(savepath))
 
			x = self.model.layer4(x)
			draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
			draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))
 
			x = self.model.avgpool(x)
			plt.plot(np.linspace(1, 2048, 2048), x.cpu().numpy()[0, :, 0, 0])
			plt.savefig("{}/f9_avgpool.png".format(savepath))
			plt.clf()
			plt.close()
 
			x = x.view(x.size(0), -1)
			x = self.model.fc(x)
			plt.plot(np.linspace(1, 1000, 1000), x.cpu().numpy()[0, :])
			plt.savefig("{}/f10_fc.png".format(savepath))
			plt.clf()
			plt.close()
		else :
			x = self.model.conv1(x)
			x = self.model.bn1(x)
			x = self.model.relu(x)
			x = self.model.maxpool(x)
			x = self.model.layer1(x)
			x = self.model.layer2(x)
			x = self.model.layer3(x)
			x = self.model.layer4(x)
			x = self.model.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.model.fc(x)
 
		return x
 
 class OpenSegNet(nn.Module):
	def __init__(self,info,id_info,args):
		super(OpenSegNet, self).__init__()
		self.ratio,self.classes = info[0],info[1]
		self.categories = id_info
		self.args = args
		self.nclass = math.floor((1-self.ratio) * 12)
		# # model
		self.model = get_segmentation_model(self.ratio,self.nclass,args.model, dataset = args.dataset,
									   backbone = args.backbone, dilated = args.dilated,
									   lateral = args.lateral, jpu = args.jpu, aux = args.aux,
									   se_loss = args.se_loss, #norm_layer = SyncBatchNorm,
									   base_size = args.base_size, crop_size = args.crop_size)
 
	def forward(self, x):
		if True: # draw features or not
			_, _, h, w = x.size()
			x = self.model.pretrained.conv1(x) #2x
			draw_features(8,8,x.cpu().numpy(),"{}/f1_conv1.png".format(savepath))
 
			x = self.model.pretrained.bn1(x) #2x
			draw_features(8, 8, x.cpu().numpy(),"{}/f2_bn1.png".format(savepath))
 
			x = self.model.pretrained.relu(x) #2x
			draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))
 
			x = self.model.pretrained.maxpool(x) #4x
			draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))
 
			c1 = self.model.pretrained.layer1(x) #4x
			draw_features(16, 16, c1.cpu().numpy(), "{}/f5_layer1.png".format(savepath))
 
			c2 = self.model.pretrained.layer2(c1) #8x
			draw_features(16, 32, c2.cpu().numpy(), "{}/f6_layer2.png".format(savepath))
 
			c3 = self.model.pretrained.layer3(c2) #8x
			draw_features(32, 32, c3.cpu().numpy(), "{}/f7_layer3.png".format(savepath))
 
			c4 = self.model.pretrained.layer4(c3) #8x
			draw_features(32, 32, c4.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
			draw_features(32, 32, c4.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))
 

			#decoder
			low_level_features = self.model.low_level(c1) #4x
			draw_features(16, 16, low_level_features.cpu().numpy(), "{}/low_level_1.png".format(savepath))
			
			x = self.model.head(c4) #8x
			draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/aspp_1.png".format(savepath))
			draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/aspp_2.png".format(savepath))
			x = F.interpolate(x,(h//4,w//4),**self._up_kwargs) #4x
			draw_features(16, 16, low_level_features.cpu().numpy(), "{}/interpolate4x.png".format(savepath))

			concated = torch.cat((low_level_features,x),1)

			objects = F.interpolate(concated,(h,w),**self._up_kwargs)
			concated = self.model.concat_conv(concated) #4x
			draw_features(16, 16, low_level_features.cpu().numpy(), "{}/interpolate4x.png".format(savepath))
			x = F.interpolate(concated, (h,w), **self._up_kwargs)
			draw_features(1, 14, low_level_features.cpu().numpy(), "{}/interpolate4x.png".format(savepath))

		else :
			x = self.model.conv1(x)
			x = self.model.bn1(x)
			x = self.model.relu(x)
			x = self.model.maxpool(x)
			x = self.model.layer1(x)
			x = self.model.layer2(x)
			x = self.model.layer3(x)
			x = self.model.layer4(x)
			x = self.model.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.model.fc(x)
 
		return x
def get_class_lists():
	data = open("logs/resnet.txt",'r').readlines()
	class_info = []
	for i in data:
		ratio,classes = i.split('|')
		ratio = float(ratio.split(':')[1])
		classes = classes.strip().split(':')[1].split(',')
		class_info.append((ratio,classes))
	return class_info
args = Options().parse()

torch.manual_seed(args.seed)
class_info = get_class_lists()
print (class_info)
root = "logs/{}".format(args.size)
if not os.path.exists(root):
	os.mkdir(root)
for i in range(1):
	id_info = Category(class_info[i][1])
	model = OpenSegNet(class_info[i],id_info,args).cuda()

# model=ft_net().cuda()
 
# pretrained_dict = resnet50.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
model.eval()
img=cv2.imread('dataset/rawdata/foregrounds/311.png')
img=cv2.resize(img,(224,224));
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
img=transform(img).cuda()
img=img.unsqueeze(0)
with torch.no_grad():
	start=time.time()
	out=model(img)
	print("total time:{}".format(time.time()-start))
	result=out.cpu().numpy()
	# ind=np.argmax(out.cpu().numpy())
	ind=np.argsort(result,axis=1)
	for i in range(5):
		print("predict:top {} = cls {} : score {}".format(i+1,ind[0,1000-i-1],result[0,1000-i-1]))
	print("done")
