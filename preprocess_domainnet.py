# -*- coding: utf-8 -*-
"""
Preprocesses DomainNet to generate h5 files 
"""
import os
import argparse
import h5py
import random
from tqdm import tqdm
import cv2
import numpy as np
from pdb import set_trace
random.seed(1)   
np.random.seed(1234)

def process_txt(txt_file):
	data = []
	with open(txt_file, "r") as f:
		data = [x.strip("\n") for x in f.readlines()]
	data = [x.split(" ") for x in data]
	data = [[x[0], int(x[1])] for x in data]
	return data

def create_dataset_for_split(domain, split, split_data, args):
	print("\nProcessing split {}".format(split))

	num_instances = len(split_data)
	im_shape = (num_instances, 224, 224, 3)
	lbl_shape = (num_instances,)
	
	save_path = os.path.join(args.output_dir, '{}_{}.h5'.format(domain, split))
	if os.path.exists(save_path): print("existed file: ", save_path);return 
	hdf5_file = h5py.File(save_path, mode="w")
	hdf5_file.create_dataset("images", im_shape)
	hdf5_file.create_dataset("labels", lbl_shape)
	
	# Store labels in dataset
	print("Adding labels..")
	hdf5_file["labels"][...] = np.array([x[1] for x in split_data])
	# Store images in dataset
	print("Adding images..")
	for i in tqdm(range(num_instances)):
		# Get path to current image
		curr_img_path = args.input_dir + split_data[i][0]
		img = cv2.imread(curr_img_path)
		img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# Save the image file
		hdf5_file["images"][i, ...] = img[None]
	hdf5_file.close()

def balance(trainval_data, val_rate=0.1):
	cla_split = {}
	for i in range(345):
		cla_split[str(i)] = []
	
	for line in trainval_data:
		cla_idx = str(line[1])
		cla_split[cla_idx].append(line)
	tr_data, val_data = [], []
	for i in range(345):
		val_num = int(len(cla_split[str(i)]) * val_rate)
		val_data.extend(cla_split[str(i)][:val_num])
		tr_data.extend(cla_split[str(i)][val_num:])
	assert len(tr_data)+len(val_data) == len(trainval_data)
	return tr_data, val_data

def generate_h5_files(domain, train_file, test_file, args):
	print('Generating train/val splits')
	trainval_data = process_txt(train_file)

	train_data, val_data = balance(trainval_data, val_rate=0.1)

	print('..done!')
	test_data = process_txt(test_file)
	print('Creating h5 files...')
	create_dataset_for_split(domain, 'train', train_data, args)
	create_dataset_for_split(domain, 'val', val_data, args)
	create_dataset_for_split(domain, 'test', test_data, args)
	print('..done!')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, default='data/domainNet/', help="Input directory")
	parser.add_argument('--domains', type=str, default='clipart,sketch', help="List of DomainNet domains to preprocess")
	parser.add_argument('--valid_ratio', type=float, default=0.1, help="Proportion of train data to be used for validation")
	parser.add_argument('--output_dir', type=str, default='./', help="Output directory")
	args = parser.parse_args()

	domains = args.domains.split(',')
	print(domains)
	# set_trace()
	for domain in domains:
		print('Processing {}...'.format(domain))		
		train_file = os.path.join(args.input_dir, '{}_train.txt'.format(domain))
		test_file = os.path.join(args.input_dir, '{}_test.txt'.format(domain))
		generate_h5_files(domain, train_file, test_file, args)

if __name__ == "__main__":
	main()