# -*- coding: utf-8 -*-
import os
import os.path as osp
import json
import random
import math
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.autograd import Function, Variable
import torch.nn.functional as F
import torchvision.transforms

from torch.utils.data.sampler import Sampler, SubsetRandomSampler

from adapt.models.models import get_model
from adapt.solvers.solver import get_solver
import yaml
from tensorboardX import SummaryWriter
import matplotlib as mpl
from sklearn.manifold import TSNE
from pdb import set_trace
import copy
from time import time
import shutil

import pynvml
pynvml.nvmlInit()

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

######################################################################
##### Miscellaneous utilities and helper classes
######################################################################


class ActualSequentialSampler(Sampler):
	r"""Samples elements sequentially, always in the same order.

	Arguments:
		data_source (Dataset): dataset to sample from
	"""

	def __init__(self, data_source):
		self.data_source = data_source

	def __iter__(self):
		return iter(self.data_source)

	def __len__(self):
		return len(self.data_source)

######################################################################
##### Training utilities
######################################################################

class ReverseLayerF(Function):
	"""
	Gradient negation utility class
	"""				 
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg()
		return output, None

class ConditionalEntropyLoss(torch.nn.Module):
	"""
	Conditional entropy loss utility class
	"""				 
	def __init__(self):
		super(ConditionalEntropyLoss, self).__init__()

	def forward(self, x):
		b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
		b = b.sum(dim=1)
		return -1.0 * b.mean(dim=0)

######################################################################
##### Sampling utilities
######################################################################

def row_norms(X, squared=False):
	"""Row-wise (squared) Euclidean norm of X.
	Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
	matrices and does not create an X.shape-sized temporary.
	Performs no input validation.
	Parameters
	----------
	X : array_like
		The input array
	squared : bool, optional (default = False)
		If True, return squared norms.
	Returns
	-------
	array_like
		The row-wise (squared) Euclidean norm of X.
	"""
	norms = np.einsum('ij,ij->i', X, X)

	if not squared:
		np.sqrt(norms, norms)
	return norms

def get_embedding(model, loader, device, num_classes, args, with_emb=False, emb_dim=512):
	# model = model.to(device)
	model.eval()
	embedding = torch.zeros([len(loader.sampler), num_classes])
	embedding_pen = torch.zeros([len(loader.sampler), emb_dim])
	labels = torch.zeros(len(loader.sampler))
	preds = torch.zeros(len(loader.sampler))
	batch_sz = args.batch_size
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(loader):
			data, target = data.to(device), target.to(device)
			if with_emb:
				try:
					e1, e2 = model(data, with_emb=True)
				except StopIteration:
					print("data.shape model.device",data.shape)
					set_trace()
				embedding_pen[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
			else:
				e1 = model(data, with_emb=False)

			embedding[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
			labels[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = target
			preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1, keepdim=True).squeeze()

	return embedding, labels, preds, embedding_pen


def train(model, device, train_loader, optimizer, epoch):
	"""
	Test model on provided data for single epoch
	"""
	model.train()
	total_loss, correct = 0.0, 0
	for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
		if data.size(0) < 2: continue
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = nn.CrossEntropyLoss()(output, target)		
		total_loss += loss.item()
		pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
		corr =  pred.eq(target.view_as(pred)).sum().item()
		correct += corr
		loss.backward()
		optimizer.step()

	train_acc = 100. * correct / len(train_loader.sampler)
	avg_loss = total_loss / len(train_loader.sampler)
	print('\nTrain Epoch: {} | Avg. Loss: {:.3f} | Train Acc: {:.3f}'.format(epoch, avg_loss, train_acc))
	return avg_loss


def test(model, device, test_loader, mode="ori", split="test", topk=1):
	"""
	Test model on provided data
	"""
	print('\nEvaluating model on {}...'.format(split))
	model.eval()
	test_loss = 0
	correct = 0
	test_acc, topk_correct = 0, 0
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			if mode == "ori":
				output = model(data)
			elif mode == "brc":
				output = model(data)[0]
			if topk>1:
				_, topk_pred = torch.topk(output, topk, dim=1)  #只支持两个的元组
				topk_target = target.unsqueeze(1).repeat(1,int(topk))
				topk_corr = topk_pred.eq(topk_target).float().sum(dim=1).sum().item()
				topk_correct += topk_corr
			loss = nn.CrossEntropyLoss()(output, target) 
			test_loss += loss.item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			corr =  pred.eq(target.view_as(pred)).sum().item()
			correct += corr
			del loss, output
			# if batch_idx%100 == 0: print("iter batch idx:", batch_idx)
	test_loss /= len(test_loader.sampler)
	test_acc = 100. * correct / len(test_loader.sampler)
	topk_acc = 100. * topk_correct / len(test_loader.sampler) if topk > 1 else -1
	return test_acc, topk_acc


######################################################################
##### Interactive visualization utilities
######################################################################

def log(target_accs, fname):
	"""
	Log results as JSON
	"""
	with open(fname, 'w') as f:
		json.dump(target_accs, f, indent=4)


def gen_dir(args):
	exp_name = '{}_{}_{}_{}r_{}b_{}'.format(args.model_init, args.al_strat, args.da_strat, \
											args.num_rounds, int(args.total_budget), args.cnn)
	
	arg_str =  'warmup{}-adapt_lr{}-wd{}'.format(args.warm_strat, args.adapt_lr, args.wd)		
	if args.da_strat == "self_ft":
		arg_str += '-srcw{:.1f}-ccw{:.1f}-ucw{:.1f}'.format(args.src_weight, args.cc_weight, args.uc_weight) 

	exps_path = osp.join('exp_record', args.round_type, args.dataset, args.id, exp_name, arg_str) 
	os.makedirs(exps_path, exist_ok=True) 
	run_num = 1 if not os.listdir(exps_path) else np.array([int(i) for i in os.listdir(exps_path) if '.txt' not in i]).max()+1

	exp_path = osp.join(exps_path, str(run_num) )
	if osp.exists(exp_path):
		set_trace();print("press c to dele exited run path: ", exp_path);shutil.rmtree(exp_path)
	os.makedirs(exp_path, exist_ok=True)
	
	with open(osp.join(exp_path, 'config.yaml'), 'w') as f:
		yaml.dump(dict(args),f)
	writer = SummaryWriter(exp_path)
	
	return writer, exp_path


def topk_feat_pred(logits, embs, cls_pro, k_feat=32, k_pred=10):
	ulb_num, emb_dim = embs.shape[0], embs.shape[1]
	_, embs_max_idx = torch.topk(embs, k_feat, dim=1)  #N,D  N,k_feat
	sort_embs_max_idx, _ = torch.sort(embs_max_idx, dim=1)
	
	_, pros_max_idx = torch.topk(cls_pro, k_feat, dim=1)  #C,D  C,k_feat
	sort_pros_max_idx, _ = torch.sort(pros_max_idx, dim=1)

	fixed_pred = torch.zeros(ulb_num)
	s_time = time()
	for i in range(ulb_num):
		# if i % 5000 == 0: print("-----now i is ",i)
		emb_i = sort_embs_max_idx[i]
		_, topk_pred_idxs = torch.topk(logits[i], k_pred)  
		candi_pros = sort_pros_max_idx[topk_pred_idxs] 
		candi_sims = torch.zeros(k_pred)
		for j in range(k_pred):
			candi_sims[j] = cal_iou(emb_i, candi_pros[j])  
		idx_in_candi = candi_sims.argmax()
		fixed_pred[i] = topk_pred_idxs[idx_in_candi]
	# print("topk_feat_pred takes mins:", (time()-s_time)//60)
	return fixed_pred

def cal_iou(a,b):
	a, b = set(a.cpu().numpy()), set(b.cpu().numpy())
	return len(a&b)/len(a|b)


def get_conf_balance(tgtuns_logits, gmm_confs, min_num, class_num):
	tgtuns_preds = torch.argmax(tgtuns_logits, dim=1)
	num_percla = min_num//class_num + 1
	while True:
		candi_idx = None  # idx in U
		for i in range(class_num):
			pred_i_idx = torch.where(tgtuns_preds==i)[0].cpu().numpy()  #idx in U
			if len(pred_i_idx) == 0: continue
			sele_idx_in_i = gmm_confs[pred_i_idx].argsort()[::-1][:num_percla]
			sele_i_idx = pred_i_idx[sele_idx_in_i].reshape(-1)
			candi_idx = sele_i_idx if candi_idx is None else np.concatenate([candi_idx, sele_i_idx])
		if len(candi_idx) >= min_num: break
		num_percla += 1
	conf_idx_in_tgtuns = np.random.choice(candi_idx, min_num, replace=False)
	return conf_idx_in_tgtuns


def get_disc(input_dim):
	disc =  nn.Sequential(
					nn.Linear(input_dim, 500),
					nn.ReLU(),
					nn.Linear(500, 500),
					nn.ReLU(),
					nn.Linear(500, 2),
					)
	return disc

def get_gpu_usedrate(need_gpu_count=1):
    used_rates = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used= float(meminfo.used/meminfo.total    )
        used_rates.append(used)
    used_num = np.array(used_rates).argsort()[:need_gpu_count]
    used_str = ','.join(map(str, used_num))
    return used_str, used_num

# for BADGE AL
def outer_product_opt(c1, d1, c2, d2):
	"""Computes euclidean distance between a1xb1 and a2xb2 without evaluating / storing cross products
	"""
	B1, B2 = c1.shape[0], c2.shape[0]
	t1 = np.matmul(np.matmul(c1[:, None, :], c1[:, None, :].swapaxes(2, 1)), np.matmul(d1[:, None, :], d1[:, None, :].swapaxes(2, 1)))
	t2 = np.matmul(np.matmul(c2[:, None, :], c2[:, None, :].swapaxes(2, 1)), np.matmul(d2[:, None, :], d2[:, None, :].swapaxes(2, 1)))
	t3 = np.matmul(c1, c2.T) * np.matmul(d1, d2.T)
	t1 = t1.reshape(B1, 1).repeat(B2, axis=1)
	t2 = t2.reshape(1, B2).repeat(B1, axis=0)
	return t1 + t2 - 2*t3

def kmeans_plus_plus_opt(X1, X2, n_clusters, init=[0], random_state=np.random.RandomState(1234), n_local_trials=None):
	"""Init n_clusters seeds according to k-means++ (adapted from scikit-learn source code)
	Parameters
	----------
	X1, X2 : array or sparse matrix
		The data to pick seeds for. To avoid memory copy, the input data
		should be double precision (dtype=np.float64).
	n_clusters : integer
		The number of seeds to choose
	init : list
		List of points already picked
	random_state : int, RandomState instance
		The generator used to initialize the centers. Use an int to make the
		randomness deterministic.
		See :term:`Glossary <random_state>`.
	n_local_trials : integer, optional
		The number of seeding trials for each center (except the first),
		of which the one reducing inertia the most is greedily chosen.
		Set to None to make the number of trials depend logarithmically
		on the number of seeds (2+log(k)); this is the default.
	Notes
	-----
	Selects initial cluster centers for k-mean clustering in a smart way
	to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
	"k-means++: the advantages of careful seeding". ACM-SIAM symposium
	on Discrete algorithms. 2007
	Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
	which is the implementation used in the aforementioned paper.
	"""

	n_samples, n_feat1 = X1.shape
	_, n_feat2 = X2.shape
	# x_squared_norms = row_norms(X, squared=True)
	centers1 = np.empty((n_clusters+len(init)-1, n_feat1), dtype=X1.dtype)
	centers2 = np.empty((n_clusters+len(init)-1, n_feat2), dtype=X1.dtype)

	idxs = np.empty((n_clusters+len(init)-1,), dtype=np.long)

	# Set the number of local seeding trials if none is given
	if n_local_trials is None:
		# This is what Arthur/Vassilvitskii tried, but did not report
		# specific results for other than mentioning in the conclusion
		# that it helped.
		n_local_trials = 2 + int(np.log(n_clusters))

	# Pick first center randomly
	center_id = init

	centers1[:len(init)] = X1[center_id]
	centers2[:len(init)] = X2[center_id]
	idxs[:len(init)] = center_id

	# Initialize list of closest distances and calculate current potential
	distance_to_candidates = outer_product_opt(centers1[:len(init)], centers2[:len(init)], X1, X2).reshape(len(init), -1)

	candidates_pot = distance_to_candidates.sum(axis=1)
	best_candidate = np.argmin(candidates_pot)
	current_pot = candidates_pot[best_candidate]
	closest_dist_sq = distance_to_candidates[best_candidate]

	# Pick the remaining n_clusters-1 points
	for c in range(len(init), len(init)+n_clusters-1):
		# Choose center candidates by sampling with probability proportional
		# to the squared distance to the closest existing center
		rand_vals = random_state.random_sample(n_local_trials) * current_pot
		candidate_ids = np.searchsorted(closest_dist_sq.cumsum(),
										rand_vals)
		# XXX: numerical imprecision can result in a candidate_id out of range
		np.clip(candidate_ids, None, closest_dist_sq.size - 1,
				out=candidate_ids)

		# Compute distances to center candidates
		distance_to_candidates = outer_product_opt(X1[candidate_ids], X2[candidate_ids], X1, X2).reshape(len(candidate_ids), -1)

		# update closest distances squared and potential for each candidate
		np.minimum(closest_dist_sq, distance_to_candidates,
				   out=distance_to_candidates)
		candidates_pot = distance_to_candidates.sum(axis=1)

		# Decide which candidate is the best
		best_candidate = np.argmin(candidates_pot)
		current_pot = candidates_pot[best_candidate]
		closest_dist_sq = distance_to_candidates[best_candidate]
		best_candidate = candidate_ids[best_candidate]

		idxs[c] = best_candidate

	return None, idxs[len(init)-1:]

if __name__ == "__main__":
    used_str, used_num = get_gpu_usedrate(need_gpu_count=1)
    print(int(used_num[0]))