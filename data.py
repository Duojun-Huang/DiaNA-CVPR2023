# -*- coding: utf-8 -*-
from hmac import trans_5C
import os
from pickle import NONE
import random
import numpy as np
import h5py

import PIL
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import utils
from pdb import set_trace
from fixmatch_data import TransformFixMatch

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

class_num = {"officehome":65, "domainnet":345, "cifar10":10 }

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset_fromlist(image_list):
	with open(image_list) as f:
		image_index = [x.split(' ')[0] for x in f.readlines()]
	with open(image_list) as f:
		label_list = []
		selected_list = []
		for ind, x in enumerate(f.readlines()):
			label = x.split(' ')[1].strip()
			label_list.append(int(label))
			selected_list.append(ind)
		image_index = np.array(image_index)
		label_list = np.array(label_list)
	image_index = image_index[selected_list]  #image index(order)
	return image_index, label_list

class DomainNetDataset(torch.utils.data.Dataset):
	def __init__(self, name, domain, split, transforms, with_strong=False):
		self.name = 'DomainNet'
		self.domain = domain
		self.split = split
		self.file_path = os.path.join('data/post_domainNet','{}_{}.h5'.format(self.domain,self.split))
		self.data, self.labels = None, None
		with h5py.File(self.file_path, 'r') as file:
			self.dataset_len = len(file["images"])
			self.num_classes = len(set(list(np.array(file['labels']))))
		self.transforms = transforms
		
		self.with_strong = with_strong
		self.strong_tranforms = TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=224)
		self.num_classes = 345

	def __len__(self):
		return self.dataset_len

	def __getitem__(self, idx):
		if self.data is None:
			self.data = h5py.File(self.file_path, 'r')["images"]
			self.labels = h5py.File(self.file_path, 'r')["labels"]
		datum, label = Image.fromarray(np.uint8(np.array(self.data[idx]))), np.array(self.labels[idx])
		if self.with_strong:
			return (self.transforms(datum), self.strong_tranforms(datum),  int(label))   #weak/strong transform  label
		return (self.transforms(datum), int(label))

	def get_num_classes(self):
		return 345


class ASDADataset:
	# Active Semi-supervised DA Dataset class
	def __init__(self, dataset, name, pair, data_dir='data', valid_ratio=0.2, batch_size=128, augment=False):
		self.dataset = dataset
		self.name = name   # domain name 
		self.pair = pair  # source/target 
		self.data_dir = data_dir
		self.valid_ratio = valid_ratio
		self.batch_size = batch_size
		self.train_size = None
		self.train_dataset = None
		self.num_classes = class_num[dataset]

	def get_num_classes(self):
		return self.num_classes

	def get_dsets(self, normalize=True, apply_transforms=True):
		if self.dataset == "domainnet":
			assert self.name in ["real", "quickdraw", "sketch", "infograph", "clipart", "painting"]
			normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
								  if normalize else transforms.Normalize([0, 0, 0], [1, 1, 1])
			if apply_transforms:
				data_transforms = {
					'train': transforms.Compose([
						transforms.Resize(256),
						transforms.RandomCrop(224),
						transforms.RandomHorizontalFlip(),  # used in CLUE for src data
						transforms.ToTensor(),
						normalize_transform
					]),
				}
			else:
				data_transforms = {
					'train': transforms.Compose([
						transforms.Resize(224),
						transforms.ToTensor(),
						normalize_transform
					]),
				}
			data_transforms['test'] = transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					normalize_transform
				])
			train_dataset = DomainNetDataset('DomainNet', self.name, 'train', data_transforms['train'])
			val_dataset = DomainNetDataset('DomainNet', self.name, 'val', data_transforms['test']) if self.pair == "source" else None
			test_dataset = DomainNetDataset('DomainNet', self.name, 'test', data_transforms['test'])
			self.num_classes = train_dataset.get_num_classes()
		else: raise NotImplementedError

		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.test_dataset = test_dataset
		
		return train_dataset, val_dataset, test_dataset

	def get_loaders(self, num_workers=4, normalize=True):
		if not self.train_dataset:
			self.get_dsets(normalize=normalize)
		
		num_train = len(self.train_dataset)
		self.train_size = num_train
		
		if self.name in ["real", "quickdraw", "sketch", "infograph", "painting", "clipart"]:
			train_idx = np.arange(len(self.train_dataset))
			train_sampler = SubsetRandomSampler(train_idx)   
		else: raise NotImplementedError

		train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler,
												batch_size=self.batch_size, num_workers=num_workers)
		val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size) if self.val_dataset is not None else None
		test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

		return train_loader, val_loader, test_loader, train_idx


if __name__ == "__main__":
	pass