


import os
import matplotlib.pyplot as plt

mode = ['test','train','val']
read_folder = "logs"
save_folder = "plots"
def plot_iteration(mode,epoch):
	target_path = os.path.join(read_folder,mode,"{}.txt".format(epoch))
	info = open(target_path,'r').readlines()
	x = list(range(len(info)))
	y = []
	for i in info:
		loss = float(i.split(',')[1].split(':')[1].strip())
		y.append(loss)
	# plt.plot(x,y)
	# plt.title("Training loss")
	# if not os.path.exists(os.path.join(save_folder,mode)):
	# 	os.mkdir(os.path.join(save_folder,mode))
	# plt.savefig(os.path.join(save_folder,mode,"{}.png".format(epoch)))
	# plt.show()
	return x,y

x1,y1 = plot_iteration('train',0)
x2,y2 = plot_iteration('train',1)
x3,y3 = plot_iteration('train',2)

x = range(len(x1+x2+x3))
y = y1+y2+y3
plt.plot(x,y)
plt.title("Training loss")
plt.savefig("training_loss.png")
plt.show()