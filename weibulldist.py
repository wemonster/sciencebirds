

import libmr

import numpy as np
import matplotlib.pyplot as plt
import torch
mr = libmr.MR()
data = np.random.randn(100)
xs = np.linspace(-5,5, 100)

# Plot two plots
# fig,(ax1,ax2) = plt.subplots(2,1)
# ax2.set_title("Fitting high")
# ax2.hist(data,bins=20,density=True)
# for tailsize in [10,30,50]:
# 	mr.fit_high(data, tailsize)
# 	assert mr.is_valid
# 	print("scale shape sign translate score  " , mr.get_params());
# 	# print("scale lb up shape lb up  " , mr.get_confidene());
# 	ax2.plot(xs, mr.w_score_vector(xs), label="Tailsize: %d"%tailsize)
# ax2.legend()

# plt.tight_layout()
# plt.show()
# for tailsize in [10]:
# 	mr.fit_high(data,tailsize)

# 	print ("Scale shape sign translate score ", mr.get_params())
# 	print ("Scale lb up shape lb up ", mr.get_confidence())
def build_weibull(features,ng=10):
	'''
	features: num_correct_samples x k
	'''
	fig, (ax1,ax2) = plt.subplots(2,1)
	ax2.set_title('Fitting high')
	weibulls = {}
	# print (features[:10,:])
	pred_class = torch.argmax(features,dim=1)
	print (pred_class.size(),features.size())
	feature_means = torch.zeros((features.size(1),1)).cuda()
	feature_sum = features.gather(1,pred_class.unsqueeze(dim=1))
	low_bound,_ = torch.min(features,dim=0)
	up_bound,_ = torch.max(features,dim=0)
	for i in range(features.size(1)):
		points = (pred_class == i).nonzero()
		feature_means[i] = torch.sum(feature_sum[points]) / points.size(0)
		weibull = libmr.MR()
		print (feature_sum[points])
		weibull.fit_high(torch.abs(feature_sum[points][:,0,0] - feature_means[i]),ng)
		xs = np.linspace(int(low_bound[i]),int(up_bound[i]),100)
		ax2.plot(xs,weibull.w_score_vector(xs))
		weibulls[i+1] = weibull
	ax2.legend()
	plt.tight_layout()
	plt.show()
	return weibulls,feature_means

feature_data = torch.load("0.pt")
weibulls,feature_means = build_weibull(feature_data)