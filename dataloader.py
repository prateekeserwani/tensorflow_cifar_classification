import numpy as np
import cv2
import os 
import random
class dataloader():
	def __init__(self,data_root,image_size,batch_size,mode, num_classes):
		self.image_size =image_size
		self.data_root = os.path.join(data_root,mode)
		self.batch_size = batch_size
		self.current_pointer= 0
		self.num_classes = num_classes
		self.mode = mode
		self.dataset = self.load_dataset()
		self.shuffle()

	def load_dataset(self):
		dataset=[]
		alist_filter = ['jpg','bmp','png','gif'] 
		for r,d,f in os.walk(self.data_root):
			for file in f:
				if file[-3:] in alist_filter:
					dataset.append([os.path.join(r,file),int(r.split('/')[-1])])
		#print(dataset)
		print('No of images ==>',len(dataset))
		return dataset

	def shuffle(self):
		for i in range(5):
			random.shuffle(self.dataset)

	def read_image(self,filename):
		image = cv2.imread(filename)
		image = cv2.resize(image,(32,32))#(self.image_size, self.image_size))
		return image
		
	def image_transform(self,image):
		image = image/255.0
		image = (image-0.5)/0.5
		return image

	def encode(self,image_label):	
		encoding = np.zeros((self.num_classes),dtype='uint8')
		encoding[image_label]=1
		return encoding 

	def create_batch(self):
		batch = np.zeros((self.batch_size, self.image_size, self.image_size,3),dtype=np.float32)
		ground_truth = np.zeros((self.batch_size, self.num_classes),dtype=np.float32)
		for index in range(self.batch_size):
			#print(self.dataset[self.current_pointer][0])
			image_name = self.dataset[self.current_pointer][0]
			image_label = self.dataset[self.current_pointer][1]
			image = self.read_image(image_name)
			image = self.image_transform(image)
			label = self.encode(image_label)
			batch[index,...]= image.copy()
			ground_truth[index,...]= label.copy()
			self.current_pointer=(self.current_pointer+1)%(len(self.dataset)-1)		
		return batch, ground_truth

'''		
data_root='./dataset/'
mode='train'
image_size=32
batch_size=5
num_classes=10	
	
print('DATASET LOADING...')		
dl = dataloader(data_root,image_size,batch_size,mode, num_classes)
for iteration in range(10000):
	image, gt = dl.create_batch()
	print(iteration, image.shape, gt.shape)
'''
			

