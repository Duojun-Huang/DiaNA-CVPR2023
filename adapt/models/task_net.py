# -*- coding: utf-8 -*-
"""
Implements individual task networks
Adapted from https://github.com/jhoffman/cycada_release
"""

import sys
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torch.nn.functional as F
from torchvision import models
import torch.nn.utils.weight_norm as weightNorm

from .models import register_model
import numpy as np
sys.path.append('../../')
import utils
from pdb import set_trace
np.random.seed(1234)
torch.manual_seed(1234)

class TaskNet(nn.Module):

    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    "Basic class which does classification."
    def __init__(self, num_cls=10, weights_init=None):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls        
        self.setup_net()        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, with_ft=False, with_emb=False, reverse_grad=False, from_emb=False):
        if from_emb:
            emb = x
        else: 
            x = self.conv_params(x)
            x = x.view(x.size(0), -1)
            x = x.clone()
            emb = self.fc_params(x)

        if isinstance(self.classifier, nn.Sequential): # LeNet
            emb = self.classifier[:-1](emb)
            if reverse_grad: emb = utils.ReverseLayerF.apply(emb)
            score = self.classifier[-1](emb)
        else:                                          # ResNet
            if reverse_grad: emb = utils.ReverseLayerF.apply(emb)
            score = self.classifier(emb)   
    
        if with_emb:
            return score, emb
        else:
            return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict, strict=False)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)
        

@register_model('ResNet34')
class ResNet34(TaskNet):
    num_channels = 3
    name = 'ResNet34'

    def setup_net(self):
        model = models.resnet34(pretrained=True)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Identity()
        
        self.classifier = nn.Linear(512, self.num_cls)
        init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()


@register_model('ssGMM')
class ss_GaussianMixture(object): 
    def __init__(self, ss_GMM_parameter=[]):
        self.ss_GMM_parameter = ss_GMM_parameter
        self.uniq = np.arange(0, len(self.ss_GMM_parameter), 1.0) #[0 1 2 3]

    def Bayes(self, X, y):
        pi = []
        sigma_dic = {}
        (n, d) = np.shape(X)
        uniq = np.unique(y)
        mu_y = np.zeros((len(uniq),d))

        for j in uniq:
            sum = 0
            for i in range(0,n,1):
                if y[i] == j:
                    sum += 1
            pi.append(sum/n)   
            
        for j in range(0,len(uniq),1):
            sum = 0
            counter = 0
            for i in range(0,n,1):
                if y[i] == uniq[j]:
                    sum = sum + X[i,:]
                    counter += 1
            mu_y[j,:] = (1/counter)*sum    
            
        for i in uniq:
            sigma_dic["SIGMA_K_" + str(i)] = np.eye(d)
            
        for j in range(0,len(uniq),1):
            sum = 0
            counter = 0
            sigma_ID = "SIGMA_K_" + str(uniq[j])
            for i in range(0,n,1):
                if y[i] == uniq[j]:
                    sum = sum + np.outer(np.transpose(X[i,:] - mu_y[j,:]),(X[i,:] - mu_y[j,:])) # (X-miu_y)*(X-miu_y)^T
                    counter += 1
            sigma_dic[sigma_ID] = (1/counter)*sum  
        
        return pi, mu_y, sigma_dic

    def gaussian_PDF(self, d, x, mu, sigma, det_sigma, sigma_inv):
        return (1/np.sqrt((2*np.pi)**(d)*det_sigma))*np.exp(-0.5*np.matmul((x-mu).T, np.matmul(sigma_inv, (x-mu))))

    def objective_func(self, beta, d, L, U, D, labeled_y, pi, mu, sigma, det_sigma, sigma_inv):
        sum_label, sum_noLabel = 0, 0
        for i in range(0,L,1):
            sigma_ID = "SIGMA_K_" + str(labeled_y[i])
            ind = self.uniq.index(labeled_y[i])
            sum_label = sum_label + np.log(pi[ind]*self.gaussian_PDF(d, D[i,:], mu[ind,:], sigma[sigma_ID], det_sigma[ind], sigma_inv[sigma_ID]))
 
        for i in range(L,L+U,1):
            inner_sum = 0
            for j in range(0,len(self.uniq),1):
                sigma_ID = "SIGMA_K_" + str(self.uniq[j])
                inner_sum = inner_sum + pi[j]*self.gaussian_PDF(d, D[i,:], mu[j,:], sigma[sigma_ID], det_sigma[j], sigma_inv[sigma_ID])
            sum_noLabel = sum_noLabel + np.log(inner_sum)

        return beta*sum_label + (1-beta)*sum_noLabel

    def fit(self, labeled_x, labeled_y, unlabeled_x, beta, tol, max_iterations, early_stop):
        cond_tolerance = 1E-10
        L = np.size(labeled_y)  
        uniq = np.unique(labeled_y)   
        K = len(uniq) 
        self.uniq = uniq.tolist()
        
        U = len(unlabeled_x)  
        D = np.concatenate((labeled_x, unlabeled_x), axis=0)  
        (n, d) = np.shape(D)  
        pi, mu, sigma = self.Bayes(labeled_x, labeled_y)   
        
        sigma_inv = {}
        det_sigma = []

        for j in range(0,len(self.uniq),1):
            sigma_ID = "SIGMA_K_" + str(self.uniq[j])
            [u, s, v] = np.linalg.svd(sigma[sigma_ID])  
            rank = len(s[s > cond_tolerance])
            det_sigma.append(np.prod(s[:rank]))  
            try:
                sigma_inv[sigma_ID] = np.linalg.pinv(sigma[sigma_ID], rcond = cond_tolerance)  
            except np.linalg.LinAlgError:
                print("The covariance matrix associated with Class " + str(self.uniq[j]) + " is still SINGULAR")
                sigma_inv[sigma_ID] = np.linalg.inv(sigma[sigma_ID], rcond = cond_tolerance)
            except:
                print("Unexpected error")
                raise 
        Objective = []
        Objective.append(self.objective_func(beta, d, L, U, D, labeled_y, pi, mu, sigma, det_sigma, sigma_inv))    
        
        GAMMA = np.zeros((n ,K))  
        obj_change = tol + 1
        t = 0
        
        while (obj_change > tol): 
            # Saving the previous GAMMA 
            GAMMA_old = np.array(GAMMA)
            for i in range(0,n,1):
                if i < L:  
                    for j in range(0,len(self.uniq),1):
                        if labeled_y[i] == self.uniq[j]:
                            GAMMA[i,j] = 1.0  
                else:
                    sum = 0
                    for j in range(0,len(self.uniq),1):
                        sigma_ID = "SIGMA_K_" + str(self.uniq[j])
                        GAMMA[i,j] = pi[j]*self.gaussian_PDF(d, D[i,:], mu[j,:], sigma[sigma_ID], det_sigma[j], sigma_inv[sigma_ID])
                        sum = sum + GAMMA[i,j]
                    GAMMA[i,:] = (1/sum)*GAMMA[i,:]  
            
            # M-STEP   
            for j in range(0,len(self.uniq),1):   
                nl = 0
                nu = 0
                for i in range(0,L,1):
                    nl = nl + GAMMA[i,j]
                for i in range(L,L+U,1):
                    nu = nu + GAMMA[i,j]
                C = (beta*nl + (1-beta)*nu)

                # Updating the cluster prior probabilities, pi        
                pi[j] = (C)/(beta*L + (1-beta)*U)
                
                # Updating the cluster means, mu
                mean_sumL = 0
                mean_sumU = 0
                for i in range(0,L,1):
                    mean_sumL = mean_sumL + GAMMA[i,j]*D[i,:]
                for i in range(L,L+U,1):
                    mean_sumU = mean_sumU + GAMMA[i,j]*D[i,:]       
                mu[j,:] = (beta*mean_sumL + (1-beta)*mean_sumU)/(C)
                
                # Updating the cluster covariance matrices, sigma
                sigma_ID = "SIGMA_K_" + str(self.uniq[j])
                
                sigma_sumL = 0
                sigma_sumU = 0
                for i in range(0,L,1):
                    sigma_sumL = sigma_sumL + GAMMA[i,j]*np.outer(np.transpose(D[i,:] - mu[j,:]),(D[i,:] - mu[j,:]))  
                for i in range(L,L+U,1):
                    sigma_sumU = sigma_sumU + GAMMA[i,j]*np.outer(np.transpose(D[i,:] - mu[j,:]),(D[i,:] - mu[j,:]))
                sigma[sigma_ID] = (beta*sigma_sumL + (1-beta)*sigma_sumU)/(C)
                
                # Updating the covariance matrix determinants and covariance inverses
                try:
                    sigma_inv[sigma_ID] = np.linalg.pinv(sigma[sigma_ID], rcond = cond_tolerance)
                    [u, s, v] = np.linalg.svd(sigma[sigma_ID])
                    rank = len(s[s>cond_tolerance])
                    det_sigma[j] = np.prod(s[:rank])
                except np.linalg.LinAlgError:
                    print("The covariance matrix associated with Class " + str(self.uniq[j]) + " has singular values, so its determinant and inverse has issues")
                    sigma_inv[sigma_ID] = np.linalg.inv(sigma[sigma_ID], rcond = cond_tolerance)
                except:
                    print("Unexpected error")
                    raise 
                
            Objective.append(self.objective_func(beta, d, L, U, D, labeled_y, pi, mu, sigma, det_sigma, sigma_inv))        
            
            if early_stop == 'True': 
                if (Objective[t] - Objective[t+1]) > 0:  
                    print('Objective function is INCREASING... stopping early and using the GAMMA from the previous iteration')
                    GAMMA = np.array(GAMMA_old)
                    break
            
            obj_change = abs((Objective[t+1] - Objective[t])/(Objective[t]))*100
            t = t + 1
            if t == max_iterations:
                print("Max number of iterations reached")
                break

        k = 0
        GMM_label_pred = np.ones(U)*99.99  
        for i in range(L,L+U,1):
            cl = GAMMA[i,:].argmax()
            GMM_label_pred[k] = self.uniq[cl]
            k = k + 1

        self.ss_GMM_parameter = []
        for j in range(0,len(self.uniq),1):
            sigma_ID = "SIGMA_K_" + str(self.uniq[j])
            self.ss_GMM_parameter.append([pi[j], mu[j,:], sigma[sigma_ID], det_sigma[j], sigma_inv[sigma_ID]])
        return self.ss_GMM_parameter

    def predict(self, x, y=None, proba=False):
        (n, d) = np.shape(x)
        GAMMA = np.zeros((n, len(self.uniq)))  #
        for i in range(n):
            sum = 0
            for j in range(0,len(self.uniq),1):
                GAMMA[i,j] = self.ss_GMM_parameter[j][0]*self.gaussian_PDF(d, x[i,:], self.ss_GMM_parameter[j][1], 
                                                            self.ss_GMM_parameter[j][2], self.ss_GMM_parameter[j][3], self.ss_GMM_parameter[j][4])
                sum = sum + GAMMA[i,j]
            GAMMA[i,:] = (1/sum)*GAMMA[i,:]

        GMM_label_pred = np.ones(n)*99.99
        for i in range(n):
            cl = GAMMA[i,:].argmax()
            GMM_label_pred[i] = self.uniq[cl]
        if y is not None:
            accuracy = np.mean(GMM_label_pred == y)
            return accuracy
        if proba:
            return GAMMA
        return GMM_label_pred