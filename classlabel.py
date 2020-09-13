class Category:
	def __init__(self,classes,unknown=False):
		self.gameObjectType = {
			'BACKGROUND':0,
			# 'UNKNOWN':1
		}

		self.id_to_cat = {
			0:'BACKGROUND',
			# 1:'UNKNOWN'
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
			self.gameObjectType[classes[i]] = i+1
			self.id_to_cat[i+1] = classes[i]
		if unknown:
			self.gameObjectType['UNKNOWN'] = len(classes)+1
			self.id_to_cat[len(classes)+1] = 'UNKNOWN'

	@property
	def ids(self):
		return self.id_to_cat.keys()
	

	def convert_class_to_category(self,class_name):
		return self.gameObjectType[class_name]

	def convert_category_to_class(self,category_id):
		return self.id_to_cat[category_id]