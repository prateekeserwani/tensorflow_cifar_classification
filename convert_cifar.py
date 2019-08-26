import pickle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

write_path ='./dataset/'

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def load_files(path):
	refine_file=[]
	files = os.listdir(path)
	for index in range(len(files)):
		if files[index].split('_')[0]=='data':
			refine_file.append(os.path.join(path,files[index])	)
	return refine_file

def write(data_dict, mode):
	image_list = data_dict[b'filenames']
	labels = data_dict[b'labels']
	image_matrix = data_dict[b'data']
	print(image_matrix.shape)
	for index in range(len(image_list)):
		image_name = image_list[index]
		image_label = labels[index]
		image = image_matrix[index]
		red = image[0:1024].reshape(32,32)
		green = image[1024:2*1024].reshape(32,32)
		blue = image[2*1024:].reshape(32,32)
		image = cv2.merge((blue,green,red))
#		plt.imshow(image)
#		plt.show()
		if not os.path.exists(os.path.join(write_path,mode, str(image_label))):
			os.mkdir(os.path.join(write_path,mode, str(image_label)))
		cv2.imwrite(os.path.join(write_path,mode,str(image_label),image_name.decode()),image)
#		input('halt')
			
	
def read_and_write(pickle_file_list):
	for index in range(len(pickle_file_list)):
		pickle_file = pickle_file_list[index];
		data_dict = unpickle(pickle_file)
		if index<len(pickle_file_list)-1:
			write(data_dict,'train')
		else:
			write(data_dict,'test')
			
			
			

pickle_list = load_files('./cifar10_original')
read_and_write(pickle_list)

#xyz = unpickle('./cifar10_original/data_batch_1')
#print(xyz[b'labels'])
#print(xyz.keys())


