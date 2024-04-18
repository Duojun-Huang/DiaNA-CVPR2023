# -*- coding: utf-8 -*-
import os
import copy
import random
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from utils import ActualSequentialSampler

from pdb import set_trace

import os
from sample import SamplingStrategy

# Alpha 
from torch.autograd import Variable
from time import time
import math

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

al_dict = {}
def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls
    return decorator

def get_sota_strategy(sample, *args):
	if sample not in al_dict: raise NotImplementedError
	try:
		return al_dict[sample](*args)
	except:
		set_trace()

@register_strategy('uniform')
class RandomSampling(SamplingStrategy):
	"""
	Uniform sampling 
	"""
	def __init__(self, dset, train_idx, model, gc_model, device, args, writer, run, exp_path):
		super(RandomSampling, self).__init__(dset, train_idx, model, gc_model, device, args, writer, run, exp_path)
		self.dset = dset

	def query(self, n):
		self.query_count += 1

		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  # The length is the length of all unlabeled target trainsets
		idx_in_unstgt = np.random.choice(np.arange(len(idxs_unlabeled)), n, replace=False)

		selected_idxs = idxs_unlabeled[idx_in_unstgt]
		return selected_idxs


@register_strategy('entropy')
class EntropySampling(SamplingStrategy):
	def __init__(self, dset, train_idx, model, gc_model, device, args, writer, run, exp_path):
		super(EntropySampling, self).__init__(dset, train_idx, model, gc_model, device, args, writer, run, exp_path)
		self.dset = dset

	def query(self, n):
		self.query_count += 1

		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  #长度为所有unlabel的target trainset的长度
		tgtuns_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		tgtuns_loader = torch.utils.data.DataLoader(self.dset, sampler=tgtuns_sampler, num_workers=4, \
												batch_size=self.args.batch_size, drop_last=False)
		##tgt unsup data emb
		tgtuns_logits , tgtuns_lab , tgtuns_preds, tgtuns_pen_emb = utils.get_embedding(self.model, tgtuns_loader, self.device, self.num_classes, \
																	self.args, with_emb=True, emb_dim=self.emb_dim)
		tgtuns_probs = F.softmax(tgtuns_logits, dim=1)
		tgtuns_entro = torch.sum(0-tgtuns_probs*torch.log(tgtuns_probs + 1e-5), dim=1).cpu().numpy()
		idx_in_unstgt = tgtuns_entro.argsort()[::-1][:n]
		
		selected_idxs = np.array(idxs_unlabeled[idx_in_unstgt])
		return selected_idxs


@register_strategy('BADGE')
class BADGESampling(SamplingStrategy):
	"""
	Implements BADGE: Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (https://arxiv.org/abs/1906.03671)
	"""
	def __init__(self, dset, train_idx, model, gc_model, device, args, writer, run, exp_path):
		super(BADGESampling, self).__init__(dset, train_idx, model, gc_model, device, args, writer, run, exp_path)
		self.dset = dset

	def query(self, n):
		self.query_count += 1
		
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, \
									batch_size=self.args.batch_size, drop_last=False)
		
		self.model.eval()

		tgt_emb = torch.zeros([len(data_loader.sampler), self.num_classes])
		tgt_pen_emb = torch.zeros([len(data_loader.sampler), self.emb_dim])
		tgt_lab = torch.zeros(len(data_loader.sampler))
		tgt_preds = torch.zeros(len(data_loader.sampler))
		batch_sz = self.args.batch_size
		
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(data_loader):
				data, target = data.to(self.device), target.to(self.device)
				e1, e2 = self.model(data, with_emb=True)
				tgt_pen_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
				tgt_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
				tgt_lab[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = target
				tgt_preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1, keepdim=True).squeeze()

		# Compute uncertainty gradient
		tgt_scores = nn.Softmax(dim=1)(tgt_emb)
		tgt_scores_delta = torch.zeros_like(tgt_scores)
		tgt_scores_delta[torch.arange(len(tgt_scores_delta)), tgt_preds.long()] = 1
		
		# Uncertainty embedding
		badge_uncertainty = (tgt_scores-tgt_scores_delta)

		# Seed with maximum uncertainty example
		max_norm = utils.row_norms(badge_uncertainty.cpu().numpy()).argmax()
		_, idx_in_unstgt = utils.kmeans_plus_plus_opt(badge_uncertainty.cpu().numpy(), tgt_pen_emb.cpu().numpy(), n, init=[max_norm])
		selected_idxs = idxs_unlabeled[idx_in_unstgt]
		return selected_idxs

@register_strategy('CLUE')
class CLUESampling(SamplingStrategy):
	"""
	Implements CLUE: CLustering via Uncertainty-weighted Embeddings
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, writer, run, exp_path):
		super(CLUESampling, self).__init__(dset, train_idx, model, discriminator, device, args, writer, run, exp_path)
		self.random_state = np.random.RandomState(1234)
		self.T = self.args.clue_softmax_t

	def query(self, n):
		self.query_count += 1
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=self.args.num_workers, \
												  batch_size=self.args.batch_size, drop_last=False)
		self.model.eval()
			
		# Get embedding of target instances   tgt_emb same shape as logits,  tgt_pen_emb embedding shape
		tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = utils.get_embedding(self.model, data_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=self.emb_dim)		
		tgt_pen_emb = tgt_pen_emb.cpu().numpy()   #unsup tgt emb
		tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)  
		tgt_scores += 1e-8
		sample_weights = -(tgt_scores*torch.log(tgt_scores)).sum(1).cpu().numpy()  #classification entropy 
		
		# Run weighted K-means over embeddings
		km = KMeans(n)
		km.fit(tgt_pen_emb, sample_weight=sample_weights)
		
		# Find nearest neighbors to inferred centroids
		dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)  # k,num_sample
		sort_idxs = dists.argsort(axis=1)  ##Sort items in K clusters
		q_idxs = []
		ax, rem = 0, n
		while rem > 0:
			q_idxs.extend(list(sort_idxs[:, ax][:rem]))  # The nearest [:, 0] is the second nearest [:, 1]
			q_idxs = list(set(q_idxs))
			rem = n-len(q_idxs)
			ax += 1

		selected_idxs = idxs_unlabeled[q_idxs]
		return selected_idxs


@register_strategy('AADA')
class AADASampling(SamplingStrategy):
	"""
	Implements Active Adversarial Domain Adaptation (https://arxiv.org/abs/1904.07848)
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		super(AADASampling, self).__init__(dset, train_idx, model, discriminator, device, args)
		self.D = None
		self.E = None

	def query(self, n):
		"""
		s(x) = frac{1-G*_d}{G_f(x))}{G*_d(G_f(x))} [Diversity] * H(G_y(G_f(x))) [Uncertainty]
		"""
		self.model.eval()
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, batch_size=64, drop_last=False)

		# Get diversity and entropy
		all_log_probs, all_scores = [], []
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(data_loader):
				data, target = data.to(self.device), target.to(self.device)
				scores = self.model(data)
				log_probs = nn.LogSoftmax(dim=1)(scores)
				all_scores.append(scores)
				all_log_probs.append(log_probs)

		all_scores = torch.cat(all_scores)
		all_log_probs = torch.cat(all_log_probs)

		all_probs = torch.exp(all_log_probs)
		disc_scores = nn.Softmax(dim=1)(self.discriminator(all_scores))
		# Compute diversity
		self.D = torch.div(disc_scores[:, 0], disc_scores[:, 1])
		# Compute entropy
		self.E = -(all_probs*all_log_probs).sum(1)
		scores = (self.D*self.E).sort(descending=True)[1]
		# Sample from top-2 % instances, as recommended by authors
		top_N = int(len(scores) * 0.02)
		q_idxs = np.random.choice(scores[:top_N].cpu().numpy(), n, replace=False)

		return idxs_unlabeled[q_idxs]

@register_strategy('Alpha')
class AlphaSampling(SamplingStrategy):
	def __init__(self, dset, train_idx, model, discriminator, device, args, writer, run, exp_path):
		super(AlphaSampling, self).__init__(dset, train_idx, model, discriminator, device, args, writer, run, exp_path)
		self.anchor = 'original'
		self.alpha_closed_form_approx  = True
		
	def query(self, n, src_loader):
		self.query_count += 1
		print('-------Run:{}/query_count:{}/ start--------'.format(self.run, self.query_count))
		
		#1.compute z* in tgt supervised and source dataset with shape[num_class,embedding_dim]
		emb_dim = self.emb_dim
		#source data emb
		src_emb, src_lab, src_preds, src_pen_emb = utils.get_embedding(self.model, src_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)
		#tgt sup data emb
		idxs_labeled = np.arange(len(self.train_idx))[self.idxs_lb]
		tgts_sampler = ActualSequentialSampler(self.train_idx[idxs_labeled])
		tgts_loader = torch.utils.data.DataLoader(self.dset, sampler=tgts_sampler, num_workers=4, \
												  batch_size=self.args.batch_size, drop_last=False)
		tgts_emb, tgts_lab, tgts_preds, tgts_pen_emb = utils.get_embedding(self.model, tgts_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)															   
		
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  
		tgtuns_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		tgtuns_loader = torch.utils.data.DataLoader(self.dset, sampler=tgtuns_sampler, num_workers=4, \
												  batch_size=self.args.batch_size, drop_last=False)
										  
		##tgt unsup data emb
		tgtuns_logits , tgtuns_lab , tgtuns_preds, tgtuns_pen_emb = utils.get_embedding(self.model, tgtuns_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)
		
		tgtuns_num = tgtuns_pen_emb.size(0)
		min_alphas = torch.ones((tgtuns_num, emb_dim), dtype=torch.float)
		candidate = torch.zeros(tgtuns_num, dtype=torch.bool)
		#2.compute the best alpha for mixing
		if self.alpha_closed_form_approx:
			tgtuns_embv = Variable(tgtuns_pen_emb, requires_grad=True).to(self.device)
			tgtuns_logits = self.model(tgtuns_embv, from_emb=True)
			tgtuns_preds = Variable(tgtuns_preds).to(self.device)
			loss = F.cross_entropy(tgtuns_logits, tgtuns_preds.long())  
			grads = torch.autograd.grad(loss, tgtuns_embv)[0].data.cpu()
			del loss, tgtuns_embv
		else:
			grads = None
		alpha_cap = 0  # Alpha_cap gradually increases, indicating that the optimal alpha is also increasing, with more drastic changes
		while alpha_cap < 1.0:
			alpha_cap +=  0.031250
			can_s = time()
			#3.compare the prediction descrepency between the mixed and original unlabeled feature and form the candidate set
			tmp_pred_change, tmp_min_alphas = self.find_candidate_set(
													src_pen_emb, tgts_pen_emb, tgtuns_pen_emb, tgtuns_preds, None, alpha_cap=alpha_cap, 
													src_Y=src_lab, Y=tgts_lab,
													grads=grads, ulb_idxs=idxs_unlabeled)  
			can_e = time()
			# print("-----------find_candidate_set used time {} mins------------".format((can_e-can_s)//60))
			is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)  
			min_alphas[is_changed] = tmp_min_alphas[is_changed]
			candidate += tmp_pred_change  #  All the indexes have been changed previously
			if self.run==0: self.writer.add_scalar("Run0/num_candidate_with_alpha_cap", int(candidate.sum().item()), alpha_cap)
			if candidate.sum() > n:
				print("alpha_cap iteration break.")
				break

		if candidate.sum() > 0:
			if self.run == 0:
				self.writer.add_scalar('Run0/stats/candidate_set_size', candidate.sum().item(), self.query_count)
				self.writer.add_scalar('Run0/stats/alpha_mean_mean', min_alphas[candidate].mean(dim=1).mean().item(), self.query_count)
				self.writer.add_scalar('Run0/stats/alpha_std_mean', min_alphas[candidate].mean(dim=1).std().item(), self.query_count)
				self.writer.add_scalar('Run0/stats/alpha_mean_std', min_alphas[candidate].std(dim=1).mean().item(), self.query_count)

			c_alpha = F.normalize(tgtuns_pen_emb[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()
			selected_idxs = self.sample(min(n, candidate.sum().item()), feats=c_alpha)  
			u_selected_idxs = candidate.nonzero(as_tuple=True)[0][selected_idxs] 
			selected_idxs = idxs_unlabeled[candidate][selected_idxs]  
		else:
			selected_idxs = np.array([], dtype=np.int)
		
		#if sample not enough
		if len(selected_idxs) < n:
			remained = n - len(selected_idxs)
			idx_lb = copy.deepcopy(self.idxs_lb)  
			idx_lb[selected_idxs] = True
			selected_idxs = np.concatenate([selected_idxs, np.random.choice(np.where(idx_lb == 0)[0], remained)])
		else: print("pick samples alpha:{} total:{}".format(len(selected_idxs) ,n))
		print('-------Run:{}/query_count:{}/ ended--------'.format(self.run, self.query_count))
		selected_idxs = np.array(selected_idxs)
		return selected_idxs


	def find_candidate_set(self, src_embedding, lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap, src_Y, Y, grads, ulb_idxs):
		unlabeled_size = ulb_embedding.size(0)
		embedding_size = ulb_embedding.size(1)

		min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float) 
		pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)
		
		if self.alpha_closed_form_approx:
			alpha_cap /= math.sqrt(embedding_size)   # epsilon = alpha_cap / math.sqrt(embedding_size)
			grads = grads.to(self.device)

		for i in range(self.num_classes):
			src_emb = src_embedding[src_Y == i]
			emb = lb_embedding[Y == i] 
			src_anchor_i = src_emb.mean(dim=0)
			tgt_anchor_i = emb.mean(dim=0)   
			assert src_emb.size(0) > 0
			if emb.size(0) > 0:
				w_src, w_tgt = src_emb.size(0)/(src_emb.size(0) + emb.size(0)), emb.size(0)/(src_emb.size(0) + emb.size(0)) 
				anchor_i = (w_src * src_anchor_i + w_tgt * tgt_anchor_i).view(1, -1).repeat(unlabeled_size, 1)  #[unlabeled_size,embed_dim]
			else:
				anchor_i = src_anchor_i.view(1, -1).repeat(unlabeled_size, 1) 
			
			
			if self.alpha_closed_form_approx:
				embed_i, ulb_embed, grads = anchor_i.to(self.device), ulb_embedding.to(self.device), grads.to(self.device)
				
				# optim_s = time()
				alpha = self.calculate_optimum_alpha(alpha_cap, embed_i, ulb_embed, grads)  #equ(5)
				# optim_e = time()
				# print("-----------used time {} mins------------".format((optim_e-optim_s)//60))
				embedding_mix = (1 - alpha) * ulb_embed + alpha * embed_i
				out = self.model(embedding_mix, from_emb=True)  #
				out = out.detach().cpu()  
				alpha = alpha.cpu()
				pc = out.argmax(dim=1) != pred_1.cpu()
			else:
				alpha = self.generate_alpha(unlabeled_size, embedding_size, alpha_cap)
				if self.args.alpha_opt:
					alpha, pc = self.learn_alpha(ulb_embedding, pred_1, anchor_i, alpha, alpha_cap,
												 log_prefix=str(i))
				else:
					embedding_mix = (1 - alpha) * ulb_embedding + alpha * anchor_i
					out, _ = self.model.clf(embedding_mix.to(self.device), embedding=True)
					out = out.detach().cpu()
					pc = out.argmax(dim=1) != pred_1

			torch.cuda.empty_cache()
			# image whose predict have not been changed alpha[idx] set to 1. shape of [unlab_num,emb_dim]
			alpha[~pc] = 1.  	
			pred_change[pc] = True   
			is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)  
			min_alphas[is_min] = alpha[is_min]  
			
		return pred_change, min_alphas


	def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
		z = (lb_embedding - ulb_embedding) #lb_embedding is some class anchor.  ulb_embedding[ulb_num,emb_dim] 
		alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8) 
		return alpha   #[ulb_num,emb_dim]

	def retrieve_anchor(self, embeddings, count):
		return embeddings.mean(dim=0).view(1, -1).repeat(count, 1)

	def generate_alpha(self, size, embedding_size, alpha_cap):
		alpha = torch.normal(
			mean=alpha_cap / 2.0,
			std=alpha_cap / 2.0,
			size=(size, embedding_size))

		alpha[torch.isnan(alpha)] = 1  

		return self.clamp_alpha(alpha, alpha_cap)  

	def clamp_alpha(self, alpha, alpha_cap):
		return torch.clamp(alpha, min=1e-8, max=alpha_cap)

	def learn_alpha(self, org_embed, labels, anchor_embed, alpha, alpha_cap, log_prefix=''):
		labels = labels.to(self.device)
		min_alpha = torch.ones(alpha.size(), dtype=torch.float) 
		pred_changed = torch.zeros(labels.size(0), dtype=torch.bool)

		loss_func = torch.nn.CrossEntropyLoss(reduction='none')

		self.model.clf.eval()

		for i in range(self.args.alpha_learning_iters):
			tot_nrm, tot_loss, tot_clf_loss = 0., 0., 0.
			for b in range(math.ceil(float(alpha.size(0)) / self.args.alpha_learn_batch_size)):
				self.model.clf.zero_grad()
				start_idx = b * self.args.alpha_learn_batch_size
				end_idx = min((b + 1) * self.args.alpha_learn_batch_size, alpha.size(0))

				l = alpha[start_idx:end_idx]
				l = Variable(l.to(self.device), requires_grad=True)
				opt = torch.optim.Adam([l], lr=self.args.alpha_learning_rate / (1. if i < self.args.alpha_learning_iters * 2 / 3 else 10.))
				e = org_embed[start_idx:end_idx].to(self.device)
				c_e = anchor_embed[start_idx:end_idx].to(self.device)
				embedding_mix = (1 - l) * e + l * c_e

				out, _ = self.model.clf(embedding_mix, embedding=True)

				label_change = out.argmax(dim=1) != labels[start_idx:end_idx]

				tmp_pc = torch.zeros(labels.size(0), dtype=torch.bool).to(self.device)
				tmp_pc[start_idx:end_idx] = label_change
				pred_changed[start_idx:end_idx] += tmp_pc[start_idx:end_idx].detach().cpu()

				tmp_pc[start_idx:end_idx] = tmp_pc[start_idx:end_idx] * (l.norm(dim=1) < min_alpha[start_idx:end_idx].norm(dim=1).to(self.device))
				min_alpha[tmp_pc] = l[tmp_pc[start_idx:end_idx]].detach().cpu()

				clf_loss = loss_func(out, labels[start_idx:end_idx].to(self.device))

				l2_nrm = torch.norm(l, dim=1)

				clf_loss *= -1

				loss = self.args.alpha_clf_coef * clf_loss + self.args.alpha_l2_coef * l2_nrm
				loss.sum().backward(retain_graph=True)
				opt.step()

				l = self.clamp_alpha(l, alpha_cap)

				alpha[start_idx:end_idx] = l.detach().cpu()

				tot_clf_loss += clf_loss.mean().item() * l.size(0)
				tot_loss += loss.mean().item() * l.size(0)
				tot_nrm += l2_nrm.mean().item() * l.size(0)

				del l, e, c_e, embedding_mix
				torch.cuda.empty_cache()

		count = pred_changed.sum().item()

		return min_alpha.cpu(), pred_changed.cpu()

	def sample(self, n, feats):
		feats = feats.numpy()
		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(feats)

		cluster_idxs = cluster_learner.predict(feats)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (feats - centers) ** 2
		dis = dis.sum(axis=1)
		return np.array(
			[np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
			 (cluster_idxs == i).sum() > 0])

