# -*- coding: utf-8 -*-
import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

import utils
from utils import ActualSequentialSampler
from adapt.solvers.solver import get_solver

from pdb import set_trace

from time import time
import os
from adapt.models.task_net import ss_GaussianMixture

from torch.cuda.amp import autocast, GradScaler
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

def get_strategy(sample, *args):
	if sample not in al_dict: raise NotImplementedError
	return al_dict[sample](*args)

class_num = {"officehome":65, "domainnet":345, "cifar10":10  }

class SamplingStrategy:
	""" 
	Sampling Strategy wrapper class
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, writer, run, exp_path):
		self.dset = dset
		self.num_classes = class_num[args.dataset]
		self.train_idx = np.array(train_idx)
		self.model = model
		if discriminator is not None:
			self.discriminator = discriminator.to(device) 
		else: 
			self.discriminator = discriminator
		self.device = device
		self.args = args
		self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)  

		# added on CLUE's code
		self.writer = writer
		self.run = run
		self.exp_path = exp_path
		self.gc_model = None
		self.query_count = 0
		self.loss_p = os.path.join(self.exp_path, 'loss')
		os.makedirs(self.loss_p, exist_ok=True)
		if self.args.cnn == 'LeNet':
			self.emb_dim = 500
		elif self.args.cnn in ['ResNet34']:
			self.emb_dim = 512
		elif self.args.cnn in ['ResNet50']:
			self.emb_dim = 2048
		elif self.args.cnn in ['ResNet50_FE']:
			self.emb_dim = 256
		else: raise NotImplementedError


	def query(self, n, src_loader):
		pass

	def update(self, idxs_lb):
		self.idxs_lb = idxs_lb
	
	def train(self, target_train_dset, args, src_loader=[], tgt_conf_loader=[], tgt_unconf_loader=[]):
		"""	
		Driver train method: using current all data to train in a semi-surpervised way
		"""

		train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])  
		
		tgt_sup_loader = torch.utils.data.DataLoader(target_train_dset, sampler=train_sampler, num_workers=args.num_workers, \
													batch_size=args.batch_size, drop_last=False)    # target lab

		tgt_unsup_loader = torch.utils.data.DataLoader(target_train_dset, shuffle=True, num_workers=args.num_workers, \
													   batch_size=args.batch_size, drop_last=False) # target lab+unlab

		opt_net_tgt = optim.Adam(self.model.parameters(), lr=args.adapt_lr, weight_decay=args.wd)
		lr_scheduler = optim.lr_scheduler.StepLR(opt_net_tgt, 20, 0.5)
		scaler = GradScaler()

		solver = get_solver(args.da_strat, self.model, src_loader, tgt_sup_loader, tgt_unsup_loader, \
						self.train_idx, opt_net_tgt, self.query_count, self.device, self.args, self.run)

		round_iter, qc_best_acc = 0, -1  # Iteration of this round (args.adapt_num_epochs epochs)

		for epoch in range(args.adapt_num_epochs):  
			if args.da_strat == 'ft':
				round_iter = solver.solve(epoch, self.writer, round_iter)
			elif args.da_strat == 'mme':
				round_iter = solver.solve(epoch, self.writer, round_iter)  
			elif args.da_strat == 'dann':
				opt_dis_adapt = optim.Adam(self.discriminator.parameters(), lr=args.adapt_lr, betas=(0.9, 0.999), weight_decay=0)
				solver.solve(epoch, self.discriminator, opt_dis_adapt)			
			elif args.da_strat == 'self_ft':
				if args.iter_num == "tgt_sup_loader":
					iter_num = len(tgt_sup_loader) 
				elif args.iter_num == "tgt_conf_loader":
					iter_num = len(tgt_conf_loader) 
				else: raise NotImplementedError
				iter_num = iter_num * args.iter_rate
				round_iter = solver.solve_common_amp(epoch, self.writer, round_iter, tgt_conf_loader, tgt_unconf_loader, iter_num, scaler,
															gmm1_train=True, conf_mask=False)
			else: raise NotImplementedError
			lr_scheduler.step()

		return self.model, qc_best_acc
	
	def get_embed(self, src_loader):
		# 1.compute z* in tgt supervised and source dataset with shape[num_class,embedding_dim]
		emb_dim = self.emb_dim
		# source data emb
		src_logits, src_lab, src_preds, src_emb = utils.get_embedding(self.model, src_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)

		# target labeled data emb
		idxs_labeled = np.arange(len(self.train_idx))[self.idxs_lb]
		tgts_sampler = ActualSequentialSampler(self.train_idx[idxs_labeled])
		tgts_loader = torch.utils.data.DataLoader(self.dset, sampler=tgts_sampler, num_workers=self.args.num_workers, \
												  batch_size=self.args.batch_size, drop_last=False)
		tgts_logits, tgts_lab, tgts_preds, tgts_emb = utils.get_embedding(self.model, tgts_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)															   
		
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  #长度为所有unlabel的target trainset的长度
		tgtuns_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		tgtuns_loader = torch.utils.data.DataLoader(self.dset, sampler=tgtuns_sampler, num_workers=self.args.num_workers, \
												  batch_size=self.args.batch_size, drop_last=False)
										  
		# target unlabeled data emb
		tgtuns_logits , tgtuns_lab , tgtuns_preds, tgtuns_emb = utils.get_embedding(self.model, tgtuns_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)
		return tgts_logits, tgts_lab, tgts_preds, tgts_emb, tgtuns_logits , tgtuns_lab , tgtuns_preds, tgtuns_emb, src_logits, src_lab, src_preds, src_emb, idxs_unlabeled


	# Calculate category-wise centroids
	def calpro_fixpred(self, src_lab,  src_pen_emb, tgtuns_logits , tgtuns_pen_emb, k_feat, tgts_lab=[], tgts_pen_emb=[]):
		emb_dim = self.emb_dim

		cls_prototypes = torch.zeros([self.num_classes, emb_dim])
		tgtuns_preds = torch.argmax(tgtuns_logits, dim=1)
		for i in range(self.num_classes):
			anchor_i = src_pen_emb[src_lab == i]
			if self.query_count > 1:
				emb = tgts_pen_emb[tgts_lab == i] 
				if len(emb) > 0: anchor_i = torch.cat([anchor_i, emb],dim=0)
			anchor_i = anchor_i.mean(dim=0).reshape(-1)
			cls_prototypes[i,:] = anchor_i
		
		fixed_unstgt_preds = utils.topk_feat_pred(tgtuns_logits, tgtuns_pen_emb, cls_prototypes, k_feat= k_feat, k_pred=self.num_classes)
		return fixed_unstgt_preds


@register_strategy('GMM')
class GMM(SamplingStrategy):
	def __init__(self, dset, train_idx, model, discriminator, device, args, writer, run, exp_path):
		super(GMM, self).__init__(dset, train_idx, model, discriminator, device, args, writer, run, exp_path)
		self.GMM_models = {}
		self.loss_type = "fix_psd"
		self.qc1_sele = True 
		self.qc_conf_type = "conf_thred" 
		self.post_conf = "max" 

	def query(self, n, src_loader):
		self.query_count += 1
		print('-------Run:{}/query_count:{}/ start--------'.format(self.run+1, self.query_count))
		
		#------GMM Training------
		#1.compute z* in tgt supervised and source dataset with shape [num_class,embedding_dim]
		emb_dim = self.emb_dim
		## source data
		src_logits, src_lab, src_preds, src_pen_emb = utils.get_embedding(self.model, src_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)												   
		## target unlabeled data 
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]  # unlabeled target trainset [0,1,..,len(U)-1]
		tgtuns_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		tgtuns_loader = torch.utils.data.DataLoader(self.dset, sampler=tgtuns_sampler, num_workers=4, batch_size = self.args.batch_size, drop_last=False)
		tgtuns_logits , tgtuns_lab , tgtuns_preds, tgtuns_pen_emb = utils.get_embedding(self.model, tgtuns_loader, self.device, \
																	   self.num_classes, self.args, with_emb=True, emb_dim=emb_dim)
																	   
		every_loss =  nn.CrossEntropyLoss(reduction="none")

		######################################################################
		##### Rule of Variant Naming
		##### -- cc(confident-consistent), uc(uncertain-consistent), ui(uncertain-inconsistent), ci(confident-inconsistent)
		##### -- S(source), T(target labeled), U(target unlabeled)
		##### -- ''*index'' means index in UST set
		######################################################################
		# STci_loss, STcc_loss, STui_loss, STuc_loss, U_loss, loss_assist_ALL = [], [], [], [], [], []
		STci_loss, STcc_loss, STui_loss, STuc_loss, U_loss = [], [], [], [], []
		loss_lab, s_time = [], time()

		ST_y = src_lab
		UST_logits = torch.cat([tgtuns_logits, src_logits], dim=0)   # 是logits
		UST_plab = torch.cat([tgtuns_preds, src_preds])  # tgtuns src都是预测的类别伪标签
		UST_label = torch.cat([tgtuns_preds, src_lab], dim=0)  
		tgts_lab, tgts_pen_emb = [], []
		if self.query_count > 1:
			idxs_labeled = np.arange(len(self.train_idx))[self.idxs_lb]
			tgts_sampler = ActualSequentialSampler(self.train_idx[idxs_labeled])
			tgts_loader = torch.utils.data.DataLoader(self.dset, sampler=tgts_sampler, num_workers=4, \
													batch_size=self.args.batch_size, drop_last=False)
			tgts_logits, tgts_lab, tgts_preds, tgts_pen_emb = utils.get_embedding(self.model, tgts_loader, self.device, self.num_classes, \
																		self.args, with_emb=True, emb_dim=emb_dim)

			ST_y = torch.cat([ST_y, tgts_lab], dim=0) #.cpu().numpy()  # Ground truth of src+stgt 
			UST_logits = torch.cat([UST_logits, tgts_logits], dim=0)  # tgtuns + src + tgts 's logits 
			UST_plab = torch.cat([UST_plab, tgts_preds]) # .cpu().numpy() 
			UST_label = torch.cat([UST_label, tgts_lab], dim=0) 

		ST_y = ST_y.cpu().numpy()
		UST_plab = UST_plab.cpu().numpy() 

		UST_y = np.concatenate([-1*np.ones(len(idxs_unlabeled)), ST_y])  # unlabel没有标签 -1,  query用真标
		UST_prob = F.softmax(UST_logits, dim=1)  # N,C 
		UST_conf = UST_prob.max(dim=1)[0].cpu().numpy()  # unlab_lab_confidence
		conf_index = np.where(UST_conf > self.args.sele_conf_thred)[0]   # model_conf > \tau  conf_index
		ST_conf_index = conf_index[conf_index > len(idxs_unlabeled)] 
		ST_uncertain_index = np.setdiff1d(np.arange(len(idxs_unlabeled), len(UST_y)), ST_conf_index)   # ST_uncertain_index
		UST_loss = every_loss(UST_logits, UST_label.long()) 

		STci_loss.extend(np.array(UST_loss)[ST_conf_index][UST_plab[ST_conf_index] != UST_y[ST_conf_index]])
		STcc_loss.extend(np.array(UST_loss)[ST_conf_index][UST_plab[ST_conf_index] == UST_y[ST_conf_index]])
		STui_loss.extend(np.array(UST_loss)[ST_uncertain_index][UST_plab[ST_uncertain_index] != UST_y[ST_uncertain_index]]) 
		STuc_loss.extend(np.array(UST_loss)[ST_uncertain_index][UST_plab[ST_uncertain_index] == UST_y[ST_uncertain_index]]) # clean pseudo label in Q

		ST_gt_loss = every_loss(UST_logits[len(idxs_unlabeled):], UST_label[len(idxs_unlabeled):].cpu().long())
		tgtuns_topkLabel = self.calpro_fixpred(src_lab, src_pen_emb, tgtuns_logits, tgtuns_pen_emb, self.args.k_feat, tgts_lab, tgts_pen_emb)  
		adapt_ploss = every_loss(tgtuns_logits, tgtuns_topkLabel.long())
		UST_loss = torch.cat([adapt_ploss, ST_gt_loss])		
		U_loss.extend(np.array(UST_loss)[np.arange(0, len(idxs_unlabeled))])
		loss_lab.extend(np.array(UST_loss)[len(idxs_unlabeled):])

		# normalization for GMM training
		UST_loss = np.array(UST_loss).reshape(-1)
		max_lossItem = max(UST_loss) # max(loss_assist_ALL) 
		min_lossItem = min(UST_loss) # min(loss_assist_ALL)
		
		STci_loss = (np.array(STci_loss) - min_lossItem) / (max_lossItem - min_lossItem)
		STcc_loss = (np.array(STcc_loss) - min_lossItem) / (max_lossItem - min_lossItem)
		STui_loss = (np.array(STui_loss) - min_lossItem) / (max_lossItem - min_lossItem)
		STuc_loss = (np.array(STuc_loss) - min_lossItem) / (max_lossItem - min_lossItem)
		U_loss = (np.array(U_loss) - min_lossItem) / (max_lossItem - min_lossItem)  # Normed unlabel data's loss
		
		x_labeled = np.concatenate([STcc_loss, STci_loss, STuc_loss, STui_loss])  # [3,2,1,0]
		y_labeled = np.concatenate([3*np.ones(len(STcc_loss)), 2*np.ones(len(STci_loss)), np.ones(len(STuc_loss)), np.zeros(len(STui_loss))])  # STcc_loss
		x_unlabeled = copy.deepcopy(U_loss)   

		s_time = time()
		m_ssGaussianMixture = ss_GaussianMixture()
		ss_GMM_parameter = m_ssGaussianMixture.fit(x_labeled.reshape(-1,1), y_labeled, x_unlabeled.reshape(-1,1), beta = 0.50, tol = 0.1, max_iterations = 20, early_stop = 'True')   
		self.GMM_models['GMM_model'] = {'ssGMM_Parameter': ss_GMM_parameter, 
										'min_loss': min_lossItem, 
										'max_loss': max_lossItem
											}
		
		ssGMM_i = m_ssGaussianMixture
		unlab_GMMprobs = ssGMM_i.predict(U_loss.reshape(-1,1), proba=True)  #[unstgt_num, 4] 
		unlab_component_conf = np.max(unlab_GMMprobs[:,0:1], axis=1)  
		idx_in_unstgt = unlab_component_conf.argsort()[::-1][:n]   # select top-n items in descending order
		assert len(idx_in_unstgt) == n

		selected_idxs = np.array(idxs_unlabeled[idx_in_unstgt])   # index in target train set

		idx_lab = np.arange(len(self.train_idx))[self.idxs_lb]
		tgtuns_mconfs = F.softmax(tgtuns_logits, dim=1).max(dim=1)[0].reshape(-1)
		min_num, min_num_gmm1 = (self.query_count + 1) * n, (self.query_count + 1) * n

		print('-----self.query_count, min_num',self.query_count, min_num)
		cc_loader, idxs3_in_unstgt = self.get_gmm_conf_loader_noactive(tgtuns_logits, idxs_unlabeled, unlab_GMMprobs, min_num, idx_in_unstgt, compnent=3)
		uc_loader, idxs1_in_unstgt = self.get_gmm_conf_loader_noactive(tgtuns_logits, idxs_unlabeled, unlab_GMMprobs, min_num_gmm1, idx_in_unstgt, compnent=1)
		
		return selected_idxs, cc_loader, uc_loader

	
	# If the amount of component is smaller than min_num, then select min_num samples according to confidence;
	# If it is larger than min_num, then select all the items in this component
	def get_gmm_conf_loader_noactive(self, U_logits, idxs_unlabeled, gmm_probs, min_num, sele_idxs_in_U, compnent=3):
		U_confs_all = gmm_probs[:,compnent].reshape(-1)

		unsele_idxs_in_U = np.setdiff1d(np.arange(len(idxs_unlabeled)), sele_idxs_in_U)
		U_logits_unsele, U_confs_unsele = U_logits[unsele_idxs_in_U], U_confs_all[unsele_idxs_in_U]

		conf_idx_in_unsele = utils.get_conf_balance(U_logits_unsele, U_confs_unsele, min_num, self.num_classes)
		conf_idx_in_U = unsele_idxs_in_U[conf_idx_in_unsele] 
		U_conf_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled[conf_idx_in_U]]) 
		fm_dataset = copy.deepcopy(self.dset) 
		if compnent == 3: fm_dataset.with_strong = True 
		U_conf_loader = torch.utils.data.DataLoader(fm_dataset, sampler=U_conf_sampler, num_workers=self.args.num_workers, \
												batch_size=self.args.batch_size, drop_last=False)
		return U_conf_loader, conf_idx_in_U 