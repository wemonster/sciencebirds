import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import encoding.dilated as resnet



class FeatureExtractor(nn.Module): # 提取特征工具
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == "fc": 
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

model = resnet.resnet50(pretrained=True) # 加载resnet50工具
model = model.cuda()
model.eval()

img=cv2.imread('dataset/characters/train/BLACKBIRD/BLACKBIRD1.png') # 加载图片

img=cv2.resize(img,(224,224));
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img=transform(img).cuda()
img=img.unsqueeze(0)

model2 = FeatureExtractor(model, ['conv1']) # 指定提取 layer3 层特征
with torch.no_grad():
    out=model2(img)
    print(len(out), out[0].shape)