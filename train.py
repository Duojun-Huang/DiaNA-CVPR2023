# -*- coding: utf-8 -*-
from math import ceil
import os
import random
import argparse
import shutil
from omegaconf import OmegaConf
import copy
import pprint
from collections import defaultdict
from tqdm import trange

import numpy as np
import torch
import pandas as pd
from pdb import set_trace
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

from adapt.models.models import get_model
import utils
from data import ASDADataset
from sample import *
from sample_sota import get_sota_strategy
import pdb
import time

from adapt.models.task_net import *

def run_active_adaptation(args, source_model, src_dset, num_classes, device, writer, exp_path, fe=None, clf=None):
	"""
	Runs active domain adaptation experiments
	"""
	# Load source data
	src_train_loader, _, _, _ = src_dset.get_loaders(num_workers=args.num_workers)   
	
	# Load target data
	target_dset = ASDADataset(args.dataset, args.target, "target", valid_ratio=0, batch_size=args.batch_size)  

	target_train_dset, _, _ = target_dset.get_dsets(apply_transforms=True)   
	
	target_train_loader, target_val_loader, target_test_loader, train_idx = target_dset.get_loaders(num_workers=args.num_workers) 

	print("len(target_train_dset):",len(target_train_dset))
	
	# Sample varying % of target data 
	if args.dataset in ['domainnet']:
		sampling_ratio = [(args.total_budget/args.num_rounds) * n for n in range(args.num_rounds+1)]   #0, 1*args.total_budget/args.num_rounds...  len args.num_rounds+1
	else: raise NotImplementedError

	# Evaluate source model on target test 
	transfer_perf, transfer_perf_topk = utils.test(source_model, device, target_test_loader, topk=1) 
	out_str = '{}->{} performance (Before {}): Task={:.2f}  Topk Acc={:.2f}'.format(args.source, args.target, args.warm_strat, transfer_perf, transfer_perf_topk)
	print(out_str)
	

	print('------------------------------------------------------\n')
	print('Running strategy: Init={}, AL={}, Pretrain={}'.format(args.model_init, args.al_strat, args.warm_strat))
	print('\n------------------------------------------------------')	

	# Choose appropriate model initialization
	if args.model_init == 'scratch':
		model, src_model = get_model(args.cnn, num_cls=num_classes).to(device), model
	elif args.model_init == 'source':
		model, src_model = source_model, source_model

	# Run unsupervised DA at round 0, where applicable
	sub_model = None # 
	if args.da_strat in ["dann"]:
		sub_model = utils.get_disc(num_classes)
	
	test_mode = "ori" 
	start_perf = transfer_perf	
	
	#################################################################
	# Main Active DA loop
	#################################################################
	target_accs = defaultdict(list)
	test_accs = np.zeros([args.runs+1 ,args.num_rounds+1])
	test_accs_gc = np.zeros([args.runs+1, args.num_rounds+1])
	test_accs_run = np.zeros([int(args.runs+1)])
	test_accs[:,0] = start_perf
	tqdm_run = trange(args.runs)
	for run in tqdm_run: # Run over multiple experimental runs
		tqdm_run.set_description('Run {}'.format(str(run)))
		tqdm_run.refresh()
		tqdm_rat = trange(len(sampling_ratio[1:]))  #len = args.num_rounds
		target_accs[0.0].append(start_perf)
		
		# Making a copy for current run
		curr_model = copy.deepcopy(model)

		# Keep track of labeled vs unlabeled data
		idxs_lb = np.zeros(len(train_idx), dtype=bool)

		# Instantiate active sampling strategy
		if args.al_strat == "GMM":
			sampling_strategy = get_strategy(args.al_strat, target_train_dset, train_idx, curr_model, sub_model, device, args, \
														writer, run, exp_path)		
		else:
			sampling_strategy = get_sota_strategy(args.al_strat, target_train_dset, train_idx, curr_model, sub_model, device, args, \
												writer, run, exp_path)		
		

		run_query_t, run_train_t = 0, 0 
		for ix in tqdm_rat: # Iterate over Active DA rounds
			ratio = sampling_ratio[ix+1]  #1
			tqdm_rat.set_description('# Target labels={:d}'.format(int(ratio)))
			tqdm_rat.refresh()

			# Select instances via AL strategy
			print('\nSelecting instances...')
			query_stime = time.time()			

			if args.al_strat == "GMM":
				idxs, uns_tgt_conf_loader, uns_tgt_unconf_loader = \
									sampling_strategy.query(int(sampling_ratio[1]), src_train_loader)
			elif args.al_strat in ["Alpha"]:
				idxs = sampling_strategy.query(int(sampling_ratio[1]), src_train_loader)
			elif args.al_strat in ["uniform", "CLUE", "entropy", "BADGE"]:
				idxs = sampling_strategy.query(int(sampling_ratio[1]))
			else: raise Exception("Not supported AL")
			
			# Record query time
			round_query_t = (time.time() - query_stime)
			run_query_t += round_query_t
			print("query of this round takes {} mins, or {:.2f} secs)".format(round_query_t//60, round_query_t ))
			
			idxs_lb[idxs] = True
			sampling_strategy.update(idxs_lb)  # update sampling_strategy.idxs_lb = idxs_lb
			if args.shuffle_src:
				src_train_loader, _, _, _ = src_dset.get_loaders(num_workers=args.num_workers) 

			# Update model with new data via DA strategy
			round_train_start = time.time()	
			if args.al_strat == "GMM": 
				best_model, qc_best_acc = sampling_strategy.train(target_train_dset, args, src_loader=src_train_loader, 
													tgt_conf_loader=uns_tgt_conf_loader, tgt_unconf_loader=uns_tgt_unconf_loader)
			else:
				best_model, qc_best_acc = sampling_strategy.train(target_train_dset, args, src_loader=src_train_loader)

			round_train_t = time.time()-round_train_start
			run_train_t += round_train_t
			print("Training of this round takes {} mins, or {:.2f} secs)".format(round_train_t//60, round_train_t ))

			# Evaluate on target test and train splits
			test_perf, test_perf_topk = utils.test(best_model, device, target_test_loader, mode=test_mode, topk=1)
			out_str = '{}->{} Test performance (Run {}  query_count {}, # Target labels={:d}): {:.2f}  Topk Acc={:.2f}'.format(args.source, args.target, run+1, \
														sampling_strategy.query_count, int(ratio), test_perf, test_perf_topk)			

			writer.add_scalar('Run{}/TargetTestAcc'.format(run), test_perf,int(ratio))

			print(out_str)
			print('\n------------------------------------------------------\n')
			
			test_accs[run, ix+1] = test_perf
			target_accs[ratio].append(test_perf)

		
		run_qt_t = run_query_t + run_train_t
		print("----- Run-{} 	 	takes {}h-{}m ({} secs)".format(run, run_qt_t//3600, (run_qt_t%3600)//60, int(run_qt_t) ))
		print("----------- query 	takes {}h-{}m ({} secs)".format(run_query_t//3600, (run_query_t%3600)//60, int(run_query_t)) )
		print("----------- training takes {}h-{}m ({} secs)".format(run_train_t//3600, (run_train_t%3600)//60, int(run_train_t)) )

		test_accs_run[run] = max(qc_best_acc, test_perf)

		# Log at the end of every run
		wargs = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
		target_accs['args'] = wargs
		utils.log(target_accs, os.path.join(exp_path,'perf.json') )
		
		if args.run_one:
			test_accs[1,:], test_accs[2,:] = test_accs[0,:], test_accs[0,:]
			test_accs_run[1:3] = max(qc_best_acc, test_perf)
			break

	test_accs[-1,:] = np.mean(test_accs[:int(args.runs),:],0)   # avg value
	accs_df = pd.DataFrame(test_accs, columns=sampling_ratio)
	accs_df.to_csv(os.path.join(exp_path,'test_accs.csv'),encoding='utf-8')
	
	test_accs_run[-1] = np.max(test_accs_run[:3])  # max value
	test_accs_run = test_accs_run.reshape(1,-1)
	test_accs_run_df = pd.DataFrame(test_accs_run, columns=['run1','run2','run3','best'])
	test_accs_run_df.to_csv(os.path.join(exp_path,'test_accs_run.csv'),encoding='utf-8')
	return target_accs

def main():
	parser = argparse.ArgumentParser()
	# Experiment identifiers
	parser.add_argument('--id', type=str, help="Experiment identifier") # transfer pair
	parser.add_argument('-a', '--al_strat', type=str, help="Active learning strategy")
	parser.add_argument('-d', '--da_strat', type=str, default='ft', help="DA strat. Currently supports: {ft, self_ft}")   #during al
	parser.add_argument('--warm_strat', type=str, default='ft', help="DA strat. Currently supports: {ft}")   #warmup
	# Load existing configuration
	parser.add_argument('--gpu',type=str, default='0', help='which gpu to use') 
	parser.add_argument('--load_from_cfg', type=bool, default=True, help="Load from config?")
	parser.add_argument('--cfg_file', type=str, help="Experiment configuration file", default="config/domainnet/clipart2sketch.yml")
	
	parser.add_argument('--thread', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=4)

	#for fix pred
	parser.add_argument('--pro_type', type=str, default="lab_mean")
	parser.add_argument('--k_feat', type=int, help='k in top-k similarity')
	parser.add_argument('--run_one', type=bool, default=True)  # one random run only
	
	#for active learning
	parser.add_argument('--round_type', type=str, default='multi-round')
	parser.add_argument('--total_budget', type=float)
	parser.add_argument('--num_rounds', type=int)

	#for training model on source domain
	parser.add_argument('--cnn', type=str)
	parser.add_argument('--warmup_epochs', type=int)

	parser.add_argument('--lr', type=float)
	parser.add_argument('--wd', type=float)

	parser.add_argument('--ft_solve', type=str, default='solve') 
	parser.add_argument('--iter_rate', type=int, default=1)
	parser.add_argument('--model_init', type=str, default="source")
	parser.add_argument('--num_epochs', type=int, default=50)

	#for adaptation on target domain
	parser.add_argument('--adapt_lr', type=float, help="DA learning rate")	# 1e-5 for dn
	parser.add_argument('--iter_num', type=str)
	parser.add_argument('--adapt_num_epochs', type=int)
	parser.add_argument('--test_best', action='store_true') 
	parser.add_argument('--batch_size', type=int)
	parser.add_argument('--shuffle_src', type=bool, default=True) 
	
	parser.add_argument('--work_root', type=str, default="./") 
	args_cmd = parser.parse_args()
	
	if args_cmd.load_from_cfg:
		args_cfg = dict(OmegaConf.load(args_cmd.cfg_file))
		args_cmd = vars(args_cmd)
		for k in args_cmd.keys():
			if args_cmd[k] is not None: args_cfg[k] = args_cmd[k]  # args_cmd as priority 
		args = OmegaConf.create(args_cfg)
	else: 
		args = args_cmd

	pp = pprint.PrettyPrinter()
	pp.pprint(args)
	torch.set_num_threads(args.thread)
	os.environ["CUDA_VISIBLE_DEVICES"] = utils.get_gpu_usedrate(1)[0]
	assert torch.cuda.is_available()
	device = torch.device(0) #use directly
	# Record
	writer, exp_path = utils.gen_dir(args)
	print("current exp_path: \t", exp_path)
	

	# Load source data
	src_dset = ASDADataset(args.dataset, args.source, "source", batch_size=args.batch_size) 
	src_train_loader, src_val_loader, src_test_loader, _ = src_dset.get_loaders(num_workers=args.num_workers)  
	num_classes = src_dset.get_num_classes()
	print('Number of classes: {}'.format(num_classes))

	# Train/load a source model
	source_model = get_model(args.cnn, num_cls=num_classes).to(device)	

	source_file = '{}_{}_source.pth'.format(args.source, args.cnn)
	source_path = os.path.join('checkpoints', 'source', args.dataset, source_file)	
	if os.path.exists(source_path): # Load existing source model
		print('Loading source checkpoint: {}'.format(source_path))
		source_model.load_state_dict(torch.load(source_path, map_location=device), strict=False)   #map location
		best_source_model = source_model
	else:							# Train source model from scarach
		print('Training {} model...'.format(args.source))
		best_val_acc, best_source_model = 0.0, None
		source_optimizer = optim.Adam(source_model.parameters(), lr=args.lr, weight_decay=args.wd)

		for epoch in range(args.num_epochs):
			utils.train(source_model, device, src_train_loader, source_optimizer, epoch)
			if (epoch+1) % 1 == 0:
				val_acc, _ = utils.test(source_model, device, src_val_loader, split="val")
				out_str = '[Epoch: {}] Val Accuracy: {:.3f} '.format(epoch, val_acc)
				print(out_str) 
				
				if (val_acc > best_val_acc):
					best_val_acc = val_acc
					best_source_model = copy.deepcopy(source_model)
					torch.save(best_source_model.state_dict(), os.path.join('checkpoints', 'source', args.dataset, source_file))

	best_source_model = torch.nn.DataParallel(best_source_model, device_ids=list(range(torch.cuda.device_count())))

	# Evaluate on source test set
	test_acc, _ = utils.test(best_source_model, device, src_test_loader, split="test")
	print('{} Test Accuracy: {:.3f} '.format(args.source, test_acc))

	# Run active adaptation experiments
	target_accs = run_active_adaptation(args, best_source_model, src_dset, num_classes, device, writer, exp_path)
	pp.pprint(target_accs)
	print("exp path: \t", exp_path)

if __name__ == '__main__':
	main()