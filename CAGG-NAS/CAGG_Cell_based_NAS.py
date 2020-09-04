#!/usr/bin/env/python
from __future__ import print_function, division

import torch
import torch.utils.data
from itertools import tee,cycle
from multiprocessing import Pool
import multiprocessing
import multiprocessing.pool
from torch import nn, optim
from torch.nn import functional as F, MSELoss, BCELoss
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Tanh, LeakyReLU, BatchNorm1d as BN1
from torch import autograd
import torch.nn.init as init
import pickle
import json
from collections import defaultdict
import networkx as nx
import numpy as np
from scipy.misc import logsumexp
import time
import random
import copy
from rdkit import Chem
import objective_func
import pylab as pl
from pytorchtools import EarlyStopping
import cma
from torch.distributions.normal import Normal
from torch_geometric.nn import global_add_pool
from torch_geometric.data import DataLoader, Data
from torch_scatter import scatter_add

import csv
import seaborn as sns
import tools

import data_utils_NASBench_201
from data_utils_NASBench_201 import get_data, to_mol, dataset_info
from nas_201_api import NASBench201API as API

#api = API('data/NAS-Bench-201-v1_0-e61699.pth')
api = API('data/NAS-Bench-201-v1_1-096897.pth')

use_prob_temp=True
use_regularize=False
num_multi_gen = 1


lam_1 = 10.
lam_2 = 1.
lam_3 = 1.

lam_4 = 0.0
alpha = 1.
eta = 0.1

SMALL_NUMBER = 1e-7 #for avoiding numerical errors
LARGE_NUMBER = 1e10

TASK_NO = 0

import argparse
parser = argparse.ArgumentParser(description='CAGG')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--dataset', help='the dataset needs to be handled ["NASBench201"] default:"NASBench201"', default="NASBench201")
parser.add_argument('--image_data', help='the image dataset needs to be handled ["cifar100", "ImageNet16-120"] default:"cifar100"', default="cifar100")
parser.add_argument('--max_nodes', type=int, default=4, metavar='S',
                    help='maximum number of nodes while generating networks (default: 4)')
parser.add_argument('--max_iter', type=int, default=390, metavar='S',
                    help='maximum number of iterations (default: 390)')
parser.add_argument('--init_num', type=int, default=10, metavar='S',
                    help='number of initialized networks (default: 10)')
parser.add_argument('--dropout', type=float, default=0.1, metavar='S',
                    help='dropout (default: 0.1)')
parser.add_argument('--acq_type', help='the type of acquisition function ["EI"] default:"EI"', default="EI")
parser.add_argument('--store_file_name', help='the output file name, default:"results/observations.csv"', default="results/observations.csv")
parser.add_argument('--pretrain_name', help='the pretrained model name', default=None) #"models/orig_rand%s"%init_num
parser.add_argument('--exist_init_nets', help='file name of the exist initialized networks', default=None) #"data/QM9_init/init_smiles_r%s.txt"%seed"
parser.add_argument('--surr_num_epochs', type=int, default=600, metavar='S',
                    help='the number of epochs in training the surrogate model (default: 600)')
parser.add_argument('--pretrain_num_epochs', type=int, default=500, metavar='S',
                    help='the number of epochs in pre-training the generation model in a VAE fashion (default: 500)')
parser.add_argument('--retrain_step', type=int, default=20, metavar='S',
                    help='every x steps to retrain the surrogate and generation models (default: 20)')
parser.add_argument('--gen_num_epochs0', type=int, default=200, metavar='S',
                    help='the number of epochs in training the generation model at the first time (default: 50)')
parser.add_argument('--gen_num_epochs1', type=int, default=10, metavar='S',
                    help='the number of epochs in retraining the generation model after the first time (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu= [int(x.split()[2]) for x in open('tmp','r').readlines()]
    free_no = str(np.argmax(memory_gpu))

device = torch.device("cuda:%s"%free_no if args.cuda else "cpu")
print("using device : ", device)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

dataset_name = args.dataset
image_data = args.image_data
max_nodes = args.max_nodes
max_iter = args.max_iter
init_num = args.init_num
dropout = args.dropout
use_random = True
acq_type = args.acq_type
use_hard = False
#property_name = args.property_name
model_name = "ours-%s-%s"%(dataset_name,image_data)
store_file_name = args.store_file_name
input_dim = 82
pretrain_name = args.pretrain_name
exist_init_nets = args.exist_init_nets
surr_num_epochs = args.surr_num_epochs
pretrain_num_epochs = args.pretrain_num_epochs
retrain_step = args.retrain_step
gen_num_epochs0 = args.gen_num_epochs0
gen_num_epochs1 = args.gen_num_epochs1


#----------------------------
#Encoder
#----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim_n, input_dim_e, em_node_mlp=[16], em_edge_mlp=[16], node_mlp=[16,16,16], edge_mlp=[16,16,16], num_fine=3, encoder_out_dim=32, dropout=0.0, encoder_act=1, device=None, Embed=None, GNBlock=None, linear_mu=None):
        super(Encoder, self).__init__()
        self.input_dim_n=input_dim_n
        self.input_dim_e=input_dim_e
        self.em_node_mlp=em_node_mlp
        self.em_edge_mlp=em_edge_mlp
        self.node_mlp=node_mlp
        self.edge_mlp=edge_mlp
        self.num_fine=num_fine
        self.encoder_out_dim=encoder_out_dim
        self.dropout=dropout
        self.device=device
      
        if encoder_act==1:
            self.encoder_act=ReLU
        else:
            self.encoder_act=Tanh

        if Embed is None:
            self.Embed = EmbedMetaLayer(EmbedEdgeModel(input_dim_e, em_edge_mlp, input_dim_n, em_node_mlp, self.encoder_act, dropout), EmbedNodeModel(input_dim_e, em_edge_mlp, input_dim_n, em_node_mlp, self.encoder_act, dropout))
        else:
            self.Embed = Embed
        
        if GNBlock is None:
            self.GNBlock = MetaLayer(EdgeModel(em_edge_mlp[-1], edge_mlp, em_node_mlp[-1], node_mlp, self.encoder_act, dropout), NodeModel(em_edge_mlp[-1], edge_mlp, em_node_mlp[-1], node_mlp, self.encoder_act, dropout))
        else:
            self.GNBlock = GNBlock

        if linear_mu is None:
            self.linear_mu = Lin(self.num_fine*(node_mlp[-1]),encoder_out_dim)
        else:
            self.linear_mu = linear_mu

        self.reset_parameters()


    def reset_parameters(self):
        for item in [self.Embed, self.GNBlock, self.linear_mu]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
            else:
                for module in item._modules.values():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr, batch, mask_edge=None):
        
        x, edge_attr = self.Embed(x, edge_index, edge_attr, batch, mask_edge)
        
        x_list = []
        for i in range(self.num_fine):
            
            x, edge_attr = self.GNBlock(x, edge_index, edge_attr, batch, mask_edge)
            
            x_list.append(global_add_pool(x, batch))
        
        inpresentation = torch.cat(x_list,dim=-1)
    
        #encoder_out_mu = self.linear_mu(inpresentation)
        encoder_out_mu = inpresentation
        #encoder_out_logvar = self.linear_logvar(inpresentation)
        #encoder_out_logvar = -torch.abs(encoder_out_logvar) #Following Mueller et al.
        
        return encoder_out_mu, inpresentation  #, encoder_out_logvar

    def soft_forward(self, x, edge_index, edge_attr, batch, weights_nodes=None, weights_edges=None):
        
        x, edge_attr = self.Embed.soft_forward(x, edge_index, edge_attr, batch, weights_nodes, weights_edges)
        
        x_list = []
        for i in range(self.num_fine):
            
            x, edge_attr = self.GNBlock.soft_forward(x, edge_index, edge_attr, batch, weights_nodes, weights_edges)
            
            x_list.append(global_add_pool(x, batch))
        
        inpresentation = torch.cat(x_list,dim=-1)
            
        #encoder_out_mu = self.linear_mu(inpresentation)
        encoder_out_mu = inpresentation
        #encoder_out_logvar = self.linear_logvar(inpresentation)
        #encoder_out_logvar = -torch.abs(encoder_out_logvar) #Following Mueller et al.
            
        return encoder_out_mu, inpresentation  #, encoder_out_logvar

    #the function is used to save the trained model
    def save_model(self, path: str):
        torch.save(self.state_dict(), "%s_encoder.pkl"%path)
    
    #the function is used to restore the trained model
    def restore_model(self, path: str):
        print("Restoring encoder from file %s." % path)
        try:
            self.load_state_dict(torch.load("%s_encoder.pkl"%path))
        except Exception as e:
            print("move the trained encoder from gpu to cpu ...")
            self.load_state_dict(torch.load("%s_encoder.pkl"%path,map_location='cpu'))


class VAEEncoder(Encoder):
    def __init__(self, input_dim_n, input_dim_e, em_node_mlp=[16], em_edge_mlp=[16], node_mlp=[16,16,16], edge_mlp=[16,16,16], num_fine=3, encoder_out_dim=32, dropout=0.0, encoder_act=1, device=None, Embed=None, GNBlock=None, linear_mu=None):
        super(VAEEncoder, self).__init__(input_dim_n, input_dim_e, em_node_mlp, em_edge_mlp, node_mlp, edge_mlp, num_fine, encoder_out_dim, dropout, encoder_act, device, Embed, GNBlock, linear_mu)
        self.linear_mu = Lin(self.num_fine*(node_mlp[-1]),encoder_out_dim)
        self.linear_logvar = Lin(self.num_fine*(node_mlp[-1]),encoder_out_dim)

    def forward(self, x, edge_index, edge_attr, batch, mask_edge=None):
        
        #print("1.2.1")
        x, edge_attr = self.Embed(x, edge_index, edge_attr, batch, mask_edge)
        #print("1.2.2")
        x_list = []
        for i in range(self.num_fine):
            
            x, edge_attr = self.GNBlock(x, edge_index, edge_attr, batch, mask_edge)
            
            x_list.append(global_add_pool(x, batch))
    
        inpresentation = torch.cat(x_list,dim=-1)
        #print("1.2.3")
        encoder_out_mu = self.linear_mu(inpresentation)
        encoder_out_logvar = self.linear_logvar(inpresentation)
        #encoder_out_logvar = -torch.abs(encoder_out_logvar) #Following Mueller et al.
        #print("1.2.4")
        return encoder_out_mu, encoder_out_logvar  #, encoder_out_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    #the function is used to save the trained model
    def save_model(self, path: str):
        torch.save(self.state_dict(), "%s_VAE_encoder.pkl"%path)
    
    #the function is used to restore the trained model
    def restore_model(self, path: str):
        print("Restoring VAE encoder from file %s." % path)
        try:
            self.load_state_dict(torch.load("%s_VAE_encoder.pkl"%path))
        except Exception as e:
            print("move the trained VAE encoder from gpu to cpu ...")
            self.load_state_dict(torch.load("%s_VAE_encoder.pkl"%path,map_location='cpu'))

#----------------------------
#Predictor
#----------------------------
class Predictor(nn.Module):
    def __init__(self, encoder_out_dim=32, sigma2_w=2.0, sigma2_noise=1e-4, mlp_pre=[50,50,50], predictor_act=2, dropout=0.0, device=None):
        super(Predictor, self).__init__()
        self.encoder_out_dim=encoder_out_dim
        self.dropout = dropout
        self.device=device
        self.mlp_pre=mlp_pre
        if predictor_act==1:
            self.predictor_act=ReLU
        else:
            self.predictor_act=Tanh

        #the linear layer of predicto
        self.linear_predictor = torch.nn.Sequential()
        predict_mlp_input_dim = encoder_out_dim
        for predict_mlp_layer_idx in range(len(self.mlp_pre)):
            self.linear_predictor.add_module("predictor_linear%s"%(predict_mlp_layer_idx+1),Lin(predict_mlp_input_dim, self.mlp_pre[predict_mlp_layer_idx]))
            #self.linear_predictor.add_module("predictor_bn%s"%(predict_mlp_layer_idx+1),BN1(self.predict_mlp_hidden))
            self.linear_predictor.add_module("predictor_tanh%s"%(predict_mlp_layer_idx+1),self.predictor_act())
            self.linear_predictor.add_module("predictor_dropout%s"%(predict_mlp_layer_idx+1),nn.Dropout(p=self.dropout))
            predict_mlp_input_dim = self.mlp_pre[predict_mlp_layer_idx]
        self.linear_predictor_readout = Lin(self.mlp_pre[-1], 1)
            
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.linear_predictor, self.linear_predictor_readout]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
            else:
                for module in item._modules.values():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

    #predictor for training
    #input: z:[B, self.encoder_out_dim]
    def forward(self, z):
        hidden = self.linear_predictor(z)
        return self.linear_predictor_readout(hidden)

    #some useful functions
    #the function is used to save the trained model
    def save_model(self, path: str):
        torch.save(self.state_dict(), "%s_predictor.pkl"%path)
    
    
    #the function is used to restore the trained model
    def restore_model(self, path: str):
        print("Restoring predictor from file %s." % path)
        try:
            self.load_state_dict(torch.load("%s_predictor.pkl"%path))
        except Exception as e:
            print("move the trained predictor from gpu to cpu ...")
            self.load_state_dict(torch.load("%s_predictor.pkl"%path,map_location='cpu'))



#------------------------------------------------------------
#DeepSurrogate
#------------------------------------------------------------
class DeepSurrogate(nn.Module):
    def __init__(self, encoder, predictor):
        super(DeepSurrogate, self).__init__()
        
        self.encoder = encoder
        self.predictor = predictor
    
        self.params = list(self.encoder.parameters())+list(self.predictor.parameters())
    
        self.reset_parameters()
    
    
    def forward(self, x, edge_index, edge_attr, batch):
        #print("(1) in surrogate")
        encoder_out_mu, _ = self.encoder(x, edge_index, edge_attr, batch)
        #print("(2) in surrogate")
        #print(encoder_out_mu.size())
        pre_mu = self.predictor(encoder_out_mu)
        #print("(3) in surrogate")
        return pre_mu
    
    def predict(self, x, edge_index, edge_attr, batch, weights_nodes=None, weights_edges=None, NSample=10, uncertainty=True, soft=False, tau=None, y_test=None):
        if uncertainty:
            self.encoder.train()
            self.predictor.train()
            res = []
            for i in range(NSample):
                if soft:
                    encoder_out_mu, _ = self.encoder.soft_forward(x, edge_index, edge_attr, batch, weights_nodes, weights_edges)
                else:
                    encoder_out_mu, _ = self.encoder(x, edge_index, edge_attr, batch, mask_edge = None)
                labels_pre = self.predictor(encoder_out_mu)
                res.append(labels_pre.view(-1,1))
            res = torch.cat(res,-1)
            pre_mu = torch.mean(res,dim=-1)
            
            if y_test is not None:
                fn = MSELoss()
                rmse = fn(y_test.view(-1),pre_mu.view(-1)).item()
                
                ll = (logsumexp(-0.5 * tau * (y_test.view(-1,1) - res.detach())**2., 1) - np.log(NSample)
                      - 0.5*np.log(2*np.pi) + 0.5*np.log(tau))
                test_ll = np.mean(ll)
            else:
                rmse, test_ll = None, None
            
            return pre_mu, torch.std(res,dim=-1), rmse, test_ll
        else:
            self.encoder.eval()
            self.predictor.eval()
            if soft:
                encoder_out_mu, _ = self.encoder.soft_forward(x, edge_index, edge_attr, batch, weights_nodes, weights_edges)
            else:
                encoder_out_mu, _ = self.encoder(x, edge_index, edge_attr, batch, mask_edge = None)
            labels_pre = self.predictor(encoder_out_mu)
            
            if y_test is not None:
                fn = MSELoss()
                rmse = fn(y_test.view(-1),labels_pre.view(-1)).item()
            
            return labels_pre, None, rmse, None
    
    def reset_parameters(self):
        for item in [self.encoder, self.predictor]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
            else:
                for module in item._modules.values():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

    
    #some useful functions
    #the function is used to save the trained model
    def save_model(self, path: str):
        self.encoder.save_model(path)
        self.predictor.save_model(path)
    
    #the function is used to restore the trained model
    def restore_model(self, path: str):
        self.encoder.restore_model(path)
        self.predictor.restore_model(path)



#------------------------------------------------------------
#Generator_DeConv
#------------------------------------------------------------
class Generator_DeConv(nn.Module):
    def __init__(self, max_nodes=30, input_dim=32, num_node_type=5, num_edge_type=3, channels = [64,32,32,1], kernels=[3,3,3,3], strides=[(1,1),(1,4),(1,2),(1,2)], paddings=[(0,0),(0,0),(0,1),(0,1)], output_padding=[(0,0),(0,0),(0,0),(0,0)],act=1, dropout=0.0, dataset="qm9", device=None):
        super(Generator_DeConv, self).__init__()
        
        self.dataset=dataset
        self.input_dim=input_dim
        self.num_node_type=num_node_type
        self.num_edge_type=num_edge_type
        self.max_nodes=max_nodes
        self.device=device
        self.channels=channels
        self.kernels=kernels
        self.strides=strides
        self.paddings=paddings
        self.output_padding=output_padding
        self.dropout=dropout
        self.use_prob_temp=use_prob_temp
        self.num_multi_gen=num_multi_gen
 

        self.params = []
    
        if act==1:
            self.act = ReLU
        elif act==2:
            self.act = Tanh
        else:
            self.act = LeakyReLU

        self.gen_model = torch.nn.Sequential()
        for i in range(len(self.channels)):
            if i==0:
                self.gen_model.add_module("ConvTranspose2d_%s"%(i+1),nn.ConvTranspose2d(self.input_dim, self.channels[i], self.kernels[i], self.strides[i], self.paddings[i], self.output_padding[i],bias=False))
            else:
                self.gen_model.add_module("ConvTranspose2d_%s"%(i+1),nn.ConvTranspose2d(self.channels[i-1], self.channels[i], self.kernels[i], self.strides[i], self.paddings[i], self.output_padding[i],bias=False))            
            if i<len(self.channels)-1:
                self.gen_model.add_module("BatchNorm2d_%s"%(i+1),nn.BatchNorm2d(self.channels[i]))
                self.gen_model.add_module("act_%s"%(i+1),self.act())
        
        self.params = self.params + list(self.gen_model.parameters())
        
        #self.reset_parameters()
        self.init_weights()
        
    
    def reset_parameters(self):
        self.init_weights()
            
    def init_weights(self):
        for p in self.params:
            p.data.normal_(0.0, 0.02)

    def forward_prob_temp(self, Z, temperature=1.0, use_random=False, use_hard=True):
    
        b = Z.size(0)
        #n = Z.size(1)
        n = self.max_nodes
        
        assert n == self.max_nodes
        #print(n,self.max_nodes)
        #assert self.input_dim == Z.size(2)
        
        #print("self.gen_model(Z.view(b,-1,1,1)).size()",self.gen_model(Z.view(b,-1,1,1)).size(),Z.view(b,-1,1,1).size())
        out = self.gen_model(Z.view(b,-1,1,1)).view(b,n,-1) #[b,1,n,(# node type + 1) + n*(# edge type + 1)]
        #out_nodes = out[:,:,0:self.num_node_type+1].view(b,n,-1) #[b,n,d]->[b, n, # node type + 1]
        
        #The following case is deterministic: the first node is input and the last node is ouput
        #Thus, we mask two dim i.e., 1,2, in the hidden nodes to 0, and mask these two dim to 1,0 
        # and 0,1 for input and ouput
        #out_nodes_mask = torch.zeros_like(out_nodes)
        #out_nodes_mask[:,:,1] = -LARGE_NUMBER
        #out_nodes_mask[:,0,1] = LARGE_NUMBER
        #out_nodes_mask[:,:,2] = -LARGE_NUMBER
        #out_nodes_mask[:,n-1,2] = LARGE_NUMBER

        #reparameterization
        #X = F.softmax(out_nodes + out_nodes_mask, dim=-1) #[b, n, # node type+1]
        
        #X is fixed, i.e., #X = tensor([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]) [b,4,5]
        X = []
        for j in range(1,self.max_nodes+1):
            node_label = data_utils_NASBench_201.onehot(j,self.max_nodes+1)
            X.append(node_label)
        X = torch.tensor(X).float().view(1,n,-1).repeat(b,1,1) #[b,4,5]
        

        all_edges = torch.triu_indices(n, n, 1).long() #[2,n*(n-1)/2]
        
        #print("out.size()=",out.size())
        out_edge = out[:,:,self.num_node_type+1:].view(b,n,n,self.num_edge_type+1) #[b, n,n, # edge type + 1]
        idx_triu = torch.triu(torch.ones(b,n,n),diagonal=1) 
        out_edge = out_edge[idx_triu==1].view(b,-1,self.num_edge_type+1) #[b,n*(n-1)/2, # edge type + 1]

        #the 0-index dim is fixed to zero
        out_edge_mask = torch.zeros_like(out_edge)
        out_edge_mask[:,:,0] = -LARGE_NUMBER
        #reparameterization
        visited_edges = F.softmax(out_edge + out_edge_mask, dim=-1) #[b, # all edge, # edge type+1]
        
        #print("visited_edges=",visited_edges)
        edge_index_one = all_edges.to(self.device)
        batch = []
        edge_index = []
        for i in range(b):
            batch = batch + [i]*n
            edge_index.append(edge_index_one+i*n)

        edge_index = torch.cat(edge_index,dim=-1).long().to(self.device).view(2,-1)
        batch = torch.tensor(batch).long().to(self.device)
        edge_attr = visited_edges
        #print("nodes=",X[:,:,1:])
        #print("active_node=",torch.sum(1.0 - X[:,:,0].view(-1)))
        #print("active_edge=",torch.sum(1.0 - edge_attr[:,:,0].view(-1)))
        #print("X=",X)

        if use_hard:
            #X_temp = torch.zeros_like(X)
            #for b_idx in range(b):
            #    for node_idx in range(n):
            #        index_one = np.random.choice(np.arange(self.num_node_type+1),p=X[b_idx,node_idx,:].detach().cpu().numpy())
            #        X_temp[b_idx,node_idx,index_one] = 1.
            X_temp = X
                    
            visited_edges_temp = torch.zeros_like(visited_edges)
            for b_idx in range(b):
                for edge_idx in range(out_edge.size(1)):
                    index_one = np.random.choice(np.arange(self.num_edge_type+1),p=visited_edges[b_idx,edge_idx,:].detach().cpu().numpy())
                    visited_edges_temp[b_idx,edge_idx,index_one] = 1.
            edge_attr_temp = visited_edges_temp
            #edge_attr_temp = torch.cat([edge_attr_temp,edge_attr_temp],dim=1) #[b, 2*alledges,# edge type+1]
            
            #compute the log probability of generation
            log_prob_temp = torch.sum(torch.sum(torch.log(X+SMALL_NUMBER) * X_temp,dim=-1),dim=-1) + torch.sum(torch.sum(torch.log(visited_edges+SMALL_NUMBER) * visited_edges_temp,dim=-1),dim=-1) #[b,]

            return [X[:,:,1:].view(-1,self.num_node_type), edge_index, edge_attr[:,:,1:].view(-1,self.num_edge_type), 1.0 - X[:,:,0].view(-1), 1.0 - edge_attr[:,:,0].view(-1), batch], [X_temp[:,:,1:].view(-1,self.num_node_type), edge_index, edge_attr_temp[:,:,1:].view(-1,self.num_edge_type), 1.0 - X_temp[:,:,0].view(-1), 1.0 - edge_attr_temp[:,:,0].view(-1), batch], log_prob_temp 
        else:
            return [X[:,:,1:].view(-1,self.num_node_type), edge_index, edge_attr[:,:,1:].view(-1,self.num_edge_type), 1.0 - X[:,:,0].view(-1), 1.0 - edge_attr[:,:,0].view(-1), batch], None, None


    def hard_decode_prob_temp(self, Z, use_random=False, decode_times=10):
        
        b = Z.size(0)
        #n = Z.size(1)
        n = self.max_nodes
        
        assert n == self.max_nodes
        #assert self.input_dim == Z.size(2)
        temperature=1.
        
        
        out = self.gen_model(Z.view(b,-1,1,1)).view(b,n,-1) #[b,1,n,(# node type + 1) + n*(# edge type + 1)]
        out_nodes = out[:,:,0:self.num_node_type+1].view(b,n,-1) #[b,n,d]->[b, n, # node type + 1]
        
        all_edges = torch.triu_indices(n, n, 1).long() #[2,n*(n-1)/2]

        out_edge = out[:,:,self.num_node_type+1:].view(b,n,n,self.num_edge_type+1) #[b, n,n, # edge type + 1]
        idx_triu = torch.triu(torch.ones(b,n,n),diagonal=1) 
        out_edge = out_edge[idx_triu==1].view(b,-1,self.num_edge_type+1) #[b,n*(n-1)/2, # edge type + 1]

        if use_random:
            X_list = []
            visited_edges_list = []
            log_prob_list = []

            #The following case is deterministic: the first node is input and the last node is ouput
            #Thus, we mask two dim i.e., 1,2, in the hidden nodes to 0, and mask these two dim to 1,0 
            # and 0,1 for input and ouput
            #out_nodes_mask = torch.zeros_like(out_nodes)
            #out_nodes_mask[:,:,1] = -LARGE_NUMBER
            #out_nodes_mask[:,0,1] = LARGE_NUMBER
            #out_nodes_mask[:,:,2] = -LARGE_NUMBER
            #out_nodes_mask[:,n-1,2] = LARGE_NUMBER

            #X = F.gumbel_softmax(out_nodes, tau=temperature, hard=True, dim=-1).detach() #[b, n, # node type+1]
            #X_probs = F.softmax(out_nodes + out_nodes_mask,dim=-1).detach() #[b, n, # node type+1]
            #X_probs is fixed, i.e., #X = tensor([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]) [b,4,5]
            X_probs = []
            for j in range(1,self.max_nodes+1):
                node_label = data_utils_NASBench_201.onehot(j,self.max_nodes+1)
                X_probs.append(node_label)
            X_probs = torch.tensor(X_probs).float().view(1,n,-1).repeat(b,1,1) #[b,4,5]
            
            #out_edge_probs = F.softmax(out_edge,dim=-1).detach() #[b, # all edge, # edge type+1]
            #the 0-index dim is fixed to zero
            out_edge_mask = torch.zeros_like(out_edge)
            out_edge_mask[:,:,0] = -LARGE_NUMBER
            #reparameterization
            out_edge_probs = F.softmax(out_edge + out_edge_mask,dim=-1).detach() #[b, # all edge, # edge type+1]

            for decode_idx in range(decode_times):
                X_temp = torch.zeros_like(X_probs)
                for b_idx in range(b):
                    for node_idx in range(n):
                        index_one = np.random.choice(np.arange(self.num_node_type+1),p=X_probs[b_idx,node_idx,:].cpu().numpy())
                        X_temp[b_idx,node_idx,index_one] = 1.
                
                visited_edges_temp = torch.zeros_like(out_edge_probs)
                for b_idx in range(b):
                    for edge_idx in range(out_edge.size(1)):
                        index_one = np.random.choice(np.arange(self.num_edge_type+1),p=out_edge_probs[b_idx,edge_idx,:].cpu().numpy())
                        visited_edges_temp[b_idx,edge_idx,index_one] = 1.

                #compute the log probability of generation
                log_prob_temp = torch.sum(torch.sum(torch.log(X_probs+SMALL_NUMBER) * X_temp,dim=-1),dim=-1) + torch.sum(torch.sum(torch.log(out_edge_probs+SMALL_NUMBER) * visited_edges_temp,dim=-1),dim=-1) #[b,]
                
                X_list.append(X_temp.view(b,1,n,-1))
                visited_edges_list.append(visited_edges_temp.view(b,1,-1,self.num_edge_type+1))
                log_prob_list.append(log_prob_temp.view(b,1))
            X_list = torch.cat(X_list,dim=1) #[b,times,n,d]
            visited_edges_list = torch.cat(visited_edges_list,dim=1) #[b,times,#edges,d]
            log_prob_list = torch.cat(log_prob_list,dim=-1) #[b,times]

            X = []
            visited_edges = []
            log_prob = []
            best_one_idx = log_prob_list.max(dim=-1)[1] #[b,]
            #print("best_one_idx=",best_one_idx)
            for i in range(b):
                X.append(X_list[i,best_one_idx[i],:,:].view(1,n,-1))
                visited_edges.append(visited_edges_list[i,best_one_idx[i],:,:].view(1,-1,self.num_edge_type+1))
                log_prob.append(log_prob_list[i,best_one_idx[i]].view(1,))

            X = torch.cat(X,dim=0)   #[b,n,d]   
            visited_edges = torch.cat(visited_edges,dim=0)   #[b,#edge,d]
            #print("log_prob=",log_prob)   
            log_prob = torch.cat(log_prob) #[b,]   

        else:
            X_list,visited_edges_list = [], []
            #The following case is deterministic: the first node is input and the last node is ouput
            #Thus, we mask two dim i.e., 1,2, in the hidden nodes to 0, and mask these two dim to 1,0 
            # and 0,1 for input and ouput
            #out_nodes_mask = torch.zeros_like(out_nodes)
            #out_nodes_mask[:,:,1] = -LARGE_NUMBER
            #out_nodes_mask[:,0,1] = LARGE_NUMBER
            #out_nodes_mask[:,:,2] = -LARGE_NUMBER
            #out_nodes_mask[:,n-1,2] = LARGE_NUMBER

            #out_nodes = out_nodes + out_nodes_mask
            #index_x = out_nodes.max(-1, keepdim=True)[1]
            #X = torch.zeros_like(out_nodes).scatter_(-1, index_x, 1.0)
            #X is fixed, i.e., #X = tensor([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]) [b,4,5]
            X = []
            for j in range(1,self.max_nodes+1):
                node_label = data_utils_NASBench_201.onehot(j,self.max_nodes+1)
                X.append(node_label)
            X = torch.tensor(X).float().view(1,n,-1).repeat(b,1,1) #[b,4,5]

            out_edge_mask = torch.zeros_like(out_edge)
            out_edge_mask[:,:,0] = -LARGE_NUMBER
            index_edge = (out_edge + out_edge_mask).max(-1, keepdim=True)[1]
            visited_edges = torch.zeros_like(out_edge).scatter_(-1, index_edge, 1.0)

            #compute the log probability of generation
            log_prob = torch.sum(torch.sum(torch.log(torch.softmax(X,dim=-1)+SMALL_NUMBER) * X,dim=-1),dim=-1) + torch.sum(torch.sum(torch.log(torch.softmax(out_edge + out_edge_mask,dim=-1)+SMALL_NUMBER) * visited_edges,dim=-1),dim=-1) #[b,]
        
    
        edge_index = all_edges.long().to(self.device).view(2,-1)
        #edge_index = edge_index[[0,1,1,0]].view(2,-1)
        edge_attr = visited_edges.to(self.device)
        #edge_attr = torch.cat([edge_attr,edge_attr],dim=1) #[b, 2*alledges,# edge type+1]
        
        
        graphs = []
        for i in range(b):
            graphs.append([X[i,:,1:], edge_index, edge_attr[i,:,1:], 1 - X[i,:,0].long().view(-1), 1 - edge_attr[i,:,0].long().view(-1),log_prob.detach()[i]])
        
        return graphs,[X_list,visited_edges_list] #[[b,times,n,d],[b,times,#edges,d]]


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    #some useful functions
    #the function is used to save the trained model
    def save_model(self, path: str):
        torch.save(self.state_dict(), "%s_generator.pkl"%path)
    
    
    #the function is used to restore the trained model
    def restore_model(self, path: str):
        print("Restoring generator from file %s." % path)
        try:
            self.load_state_dict(torch.load("%s_generator.pkl"%path))
        except Exception as e:
            print("move the trained generator from gpu to cpu ...")
            self.load_state_dict(torch.load("%s_generator.pkl"%path,map_location='cpu'))




#------------------------------------------------------------
#Train the VAE
#------------------------------------------------------------
class TrainVAE():
    def __init__(self, VAEencoder, generator, surrogate=None):
        self.VAEencoder = VAEencoder
        self.generator = generator
        self.surrogate = surrogate
        self.use_prob_temp = use_prob_temp
        self.lam_2 = lam_2
        self.lam_3 = lam_3

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def train(self, data, batch_size=32, num_epochs=100, learning_rate=1e-3, weight_decay=1e-5, temperature=1.0, run_one_batch=False):

        optimizer = torch.optim.Adam(list(self.VAEencoder.parameters())+self.generator.params, lr=learning_rate, weight_decay=weight_decay)

        loss_store={"total_loss":[],"elbo_loss":[]}
        train_time=time.time()
        for epoch in range(0, num_epochs):
            
            optimizer.zero_grad()
            
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
            step = 0
            processed_graphs = 0
            
            for batch_data in data_loader:
                step += 1
                
                processed_graphs += batch_data.num_graphs
                
                optimizer.zero_grad()
                
                encoder_out_mu, encoder_out_logvar = self.VAEencoder(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, mask_edge = None)
            
                #the loss of encoder
                # see Appendix B from VAE paper:
                # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                # https://arxiv.org/abs/1312.6114
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                loss_kl = -0.5 * torch.sum(1 + encoder_out_logvar - encoder_out_mu.pow(2) - encoder_out_logvar.exp(),-1) #[b,]

                #reconstruction loss
                z = self.reparameterize(encoder_out_mu, encoder_out_logvar)
                #print(z.size(),encoder_out_mu.size(),encoder_out_logvar.size())
                [nodes_labels, edge_index, edge_attr, active_nodes, active_edges, batch], _, _ = self.generator.forward_prob_temp(z, temperature=temperature, use_random=use_random,use_hard=False)
                #print(nodes_labels.size(),active_nodes.size())
                fake_x = torch.cat([1.0-active_nodes.view(-1,1),nodes_labels],dim=-1).view(batch_size,-1,self.generator.num_node_type+1) #[b,n,1+d]
                fake_attr = torch.cat([1.0-active_edges.view(-1,1),edge_attr],dim=-1).view(batch_size,-1,self.generator.num_edge_type+1) #[b,n*(n-1),1+d]
                
                real_x = tools.to_dense_batch(batch_data.x, max_num_nodes=self.generator.max_nodes, batch=batch_data.batch, fill_value=0).view(batch_size,-1,self.generator.num_node_type+1) #[b,n,1+d]
                real_attr = tools.to_dense_adj(batch_data.edge_index, max_num_nodes=self.generator.max_nodes, batch=batch_data.batch, edge_attr=batch_data.edge_attr).view(batch_size,-1,self.generator.num_edge_type+1) #[b,n*(n-1),1+d]

                loss_rec = torch.sum(torch.sum(torch.log(fake_x+SMALL_NUMBER) * real_x,dim=-1),dim=-1) + torch.sum(torch.sum(torch.log(fake_attr+SMALL_NUMBER) * real_attr,dim=-1),dim=-1)
                #loss_rec = torch.prod(torch.prod(torch.pow(fake_x,real_x),dim=-1),dim=-1) * torch.prod(torch.prod(torch.pow(fake_attr,real_attr),dim=-1),dim=-1) #[b,]
                #loss_rec = (loss_rec+SMALL_NUMBER).log()
                
                loss_elbo = (loss_kl - loss_rec).mean()
                
                
                #reg
                Z = torch.randn(batch_size,self.generator.input_dim)
                [nodes_labels, edge_index, edge_attr, active_nodes, active_edges, batch],_,_ = self.generator.forward_prob_temp(Z, temperature=temperature, use_random=use_random,use_hard=False)
                #the strong connectivity constraint
                row, col = edge_index
                a = -100.
                A = torch.sparse.FloatTensor(edge_index,active_edges,torch.Size([batch_size*self.generator.max_nodes,batch_size*self.generator.max_nodes])).to_dense()
                #print("A=",A)
                A_list = [torch.eye(batch_size*self.generator.max_nodes), A]
                #print("A[row, col]=",A[row, col],A[row, col].sum())
                for _ in range(2,self.generator.max_nodes):
                    A_mm = torch.mm(A_list[-1],A)
                    #print("A_mm[row, col]=",A_mm[row, col],A_mm[row, col].sum())
                    A_temp = torch.zeros_like(A_mm)
                    A_temp[row, col] = 1. / (1. + (a * ( A_mm[row, col] - 0.5)).exp())
                    #print("A_temp=",A_temp)
                    #print("A_temp[row, col]=",A_temp[row, col],A_temp[row, col].sum())
                    A_list.append(A_temp)
                                
                sum_A = sum(A_list)
                C = 1. / (1. + (a * (sum_A[row, col] - 0.5)).exp())
                
                #print("C=",C,C.sum())

                x_idx = torch.arange(batch_size*self.generator.max_nodes).long()
                input_idx = (torch.arange(0,batch_size*self.generator.max_nodes,self.generator.max_nodes).float().view(batch_size,1)*torch.ones(batch_size,self.generator.max_nodes)).view(-1).long()
                output_idx = (torch.arange(self.generator.max_nodes-1,batch_size*self.generator.max_nodes,self.generator.max_nodes).float().view(batch_size,1)*torch.ones(batch_size,self.generator.max_nodes)).view(-1).long()

                C_input = 1. / (1. + (a * (sum_A[input_idx, x_idx] - 0.5)).exp())
                C_output = 1. / (1. + (a * (sum_A[x_idx, output_idx] - 0.5)).exp())
                #print(active_nodes.view(-1))
                #print(sum(C_input),sum(C_output))

                loss_reg_connect_indeg = active_nodes.view(-1) * (1. - C_input) + (1. - active_nodes.view(-1)) * scatter_add(C, col, dim=0, out=torch.zeros(batch_size*self.generator.max_nodes))
                loss_reg_connect_outdeg = active_nodes.view(-1) * (1. - C_output) + (1. - active_nodes.view(-1)) * scatter_add(C, row, dim=0, out=torch.zeros(batch_size*self.generator.max_nodes))
                #print("scatter_add(C, col, dim=0, out=torch.zeros(batch_size*self.generator.max_nodes)=",scatter_add(C, col, dim=0, out=torch.zeros(batch_size*self.generator.max_nodes)))
                #print("scatter_add(C, row, dim=0, out=torch.zeros(batch_size*self.generator.max_nodes)=",scatter_add(C, row, dim=0, out=torch.zeros(batch_size*self.generator.max_nodes)))

                loss_reg_connect = loss_reg_connect_indeg[loss_reg_connect_indeg>SMALL_NUMBER].sum()

                loss_reg_valid = loss_reg_connect_outdeg[loss_reg_connect_outdeg>SMALL_NUMBER].sum()
                
                loss = loss_elbo + self.lam_2 * loss_reg_valid + self.lam_3 * loss_reg_connect
                #loss = loss_elbo + loss_reg_valid + loss_reg_connect

                loss.backward()

                optimizer.step()
            
                if run_one_batch:
                    print("training vae : epoch %s, total loss: %.5f, elbo: %.5f [kl: %.5f, recon: %.5f], connect: %.5f, valid: %.5f | time cost: %.2f" % (epoch+1,loss.item(),loss_elbo.item(),loss_kl.mean().item(),-loss_rec.mean().item(),loss_reg_connect.item(),loss_reg_valid.item(), time.time()-train_time),end="\n")
                    return

            loss_store["total_loss"].append(loss.item())
            loss_store["elbo_loss"].append(loss_elbo.item())

            print("training vae : epoch %s, total loss: %.5f, elbo: %.5f [kl: %.5f, recon: %.5f], connect: %.5f, valid: %.5f | time cost: %.2f" % (epoch+1,loss.item(),loss_elbo.item(),loss_kl.mean().item(),-loss_rec.mean().item(),loss_reg_connect.item(),loss_reg_valid.item(), time.time()-train_time),end="\n")


            pl.plot(loss_store["total_loss"],'k')
            pl.plot(loss_store["elbo_loss"],'r')
            pl.legend(["total_loss","elbo_loss"])
            #pl.savefig("vae_log_loss.png")
            pl.close()




#------------------------------------------------------------
#Train the Generator
#------------------------------------------------------------
class TrainGenerator():
    def __init__(self, generator=None, surrogate=None, discriminator=None, lam_1=None, lam_2=None, lam_3=None):
        self.generator = None
        self.surrogate = surrogate
        self.discriminator = discriminator
        self.use_prob_temp = use_prob_temp
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.lam_3 = lam_3
        self.lam_4 = lam_4
        self.alpha = alpha
        self.eta = eta
        self.reparam_num = 1
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def construct_loss(self, surrogate, nodes_labels, edge_index, edge_attr, active_nodes, active_edges, batch, temperature=1.0, use_random=False, log_rec_loss=None):
        
        pre_mu, pre_sigma2,_,_ = surrogate.predict(nodes_labels, edge_index, edge_attr, batch, weights_nodes=active_nodes, weights_edges=active_edges, NSample=10, uncertainty=True, soft=True, tau=None, y_test=None)
        #print("pre_mu=",pre_mu)
        
        #loss = torch.mean(pre_mu+self.eta*torch.sqrt(pre_sigma2))
        #loss = torch.mean(pre_mu-self.eta*torch.sqrt(pre_sigma2))
        b = pre_mu.size(0)
        #print("log_rec_loss=",log_rec_loss.size(),"mean.size=",torch.mean(self.reparameterize(pre_mu.view(b,1,-1).repeat(1,self.reparam_num,1), (pre_sigma2.view(b,1,-1).repeat(1,self.reparam_num,1)+SMALL_NUMBER).log()),dim=1).size())
        if log_rec_loss is None:
            loss = torch.mean(torch.mean(self.reparameterize(pre_mu.view(b,1,-1).repeat(1,self.reparam_num,1), (pre_sigma2.view(b,1,-1).repeat(1,self.reparam_num,1)+SMALL_NUMBER).log()),dim=1),dim=0) #[b,reparam_num,-1]->[b,-1]->[-1,]
        else:
            loss = torch.mean(log_rec_loss.view(b,-1) * torch.mean(self.reparameterize(pre_mu.view(b,1,-1).repeat(1,self.reparam_num,1), (pre_sigma2.view(b,1,-1).repeat(1,self.reparam_num,1)+SMALL_NUMBER).log()),dim=1),dim=0) #[b,reparam_num,-1]->[b,-1]->[-1,]
        #print("loss=",loss)
        return -loss                

    def train_GAN_and_Exp(self, generator, real_data, batch_size=10, batch_size_vae=100, lr_gen=1e-4, lr_disc=1e-4, weight_decay=0.0, temperature=1.0, temperature_annealing_rate=None, num_epochs_gen=50, num_epochs_disc=5, use_random=False, use_regularize=True, use_exp=True, vaeencoder=None, surrogate=None, params=None):
        
        data_set_copy = real_data
        batch_size_vae = min(int(len(data_set_copy)/2),batch_size_vae) #choose best 1/2 to train 
        #sort data
        data_set_copy.data_list.sort(key=lambda d:d.y[:],reverse=True)
        #data_set_copy.cut(0,min(len(data_set_copy)/2,batch_size_vae)) #choose best 1/2 to train 

        total_time_start = time.time()
        
        loss_store = defaultdict(list)
        
        loss_opt = LARGE_NUMBER
        epoch_opt = -1
        tau = temperature
        
        clamp_lower=-0.5
        clamp_upper=0.5
        
        if surrogate is None:
            surrogate = self.surrogate
        
        #do not compute the grad of parameters in encoder and predictor
        for p in surrogate.encoder.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.data.zero_()
            p.requires_grad=False
        for p in surrogate.predictor.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.data.zero_()
            p.requires_grad=False
        
        if vaeencoder is None:
            optimizer_gen = torch.optim.Adam(generator.params, lr=lr_gen, weight_decay=weight_decay)
        else:
            optimizer_gen = torch.optim.Adam(list(vaeencoder.parameters())+generator.params, lr=lr_gen, weight_decay=weight_decay)
        
        #print("1.1")
        for epoch in range(0, num_epochs_gen):
            
            optimizer_gen.zero_grad()
            
            loss_d = torch.tensor([0.])
            loss_g = torch.tensor([0.])
            
            optimizer_gen.zero_grad()
            
            if vaeencoder is not None:
            
                #real data
                data_loader = DataLoader(data_set_copy, batch_size=batch_size_vae, shuffle=False)
  
                dl_data = iter(data_loader)

                batch_data = next(dl_data)
                #print("1.2")
                encoder_out_mu, encoder_out_logvar = vaeencoder(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, mask_edge = None)
                #print("1.3")
                #the loss of encoder
                # see Appendix B from VAE paper:
                # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                # https://arxiv.org/abs/1312.6114
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                loss_kl = -0.5 * torch.sum(1 + encoder_out_logvar - encoder_out_mu.pow(2) - encoder_out_logvar.exp(),-1) #[b,]
                
                #reconstruction loss
                z = vaeencoder.reparameterize(encoder_out_mu, encoder_out_logvar)
                #print(z.size(),encoder_out_mu.size(),encoder_out_logvar.size())
                [nodes_labels, edge_index, edge_attr, active_nodes, active_edges, batch], _, _ = generator.forward_prob_temp(z, temperature=temperature, use_random=use_random, use_hard=False)
                fake_x = torch.cat([1.0-active_nodes.view(-1,1),nodes_labels],dim=-1).view(batch_size_vae,-1,generator.num_node_type+1) #[b,n,1+d]
                fake_attr = torch.cat([1.0-active_edges.view(-1,1),edge_attr],dim=-1).view(batch_size_vae,-1,generator.num_edge_type+1) #[b,n*(n-1),1+d]
                
                real_x = tools.to_dense_batch(batch_data.x, max_num_nodes=generator.max_nodes, batch=batch_data.batch, fill_value=0).view(batch_size_vae,-1,generator.num_node_type+1) #[b,n,1+d]
                real_attr = tools.to_dense_adj(batch_data.edge_index, max_num_nodes=generator.max_nodes, batch=batch_data.batch, edge_attr=batch_data.edge_attr).view(batch_size_vae,-1,generator.num_edge_type+1) #[b,n*(n-1),1+d]
                
                loss_rec = torch.sum(torch.sum(torch.log(fake_x+SMALL_NUMBER) * real_x,dim=-1),dim=-1) + torch.sum(torch.sum(torch.log(fake_attr+SMALL_NUMBER) * real_attr,dim=-1),dim=-1)
                #loss_rec = torch.prod(torch.prod(torch.pow(fake_x,real_x),dim=-1),dim=-1) * torch.prod(torch.prod(torch.pow(fake_attr,real_attr),dim=-1),dim=-1) #[b,]
                #loss_rec = (loss_rec+SMALL_NUMBER).log()
                
                loss_elbo = (loss_kl - loss_rec).mean()
            
            else:
                loss_elbo = torch.tensor([0.])
            #print("1.4")
            #dealing with fake data
            #Z = torch.randn(batch_size,generator.max_nodes,generator.input_dim)
            Z = torch.randn(batch_size,generator.input_dim)
            if self.use_prob_temp:
                [nodes_labels, edge_index, edge_attr, active_nodes, active_edges, batch],temp_hard,recon_loss = generator.forward_prob_temp(Z, temperature=temperature, use_random=use_random, use_hard=use_hard)
            else:
                nodes_labels, edge_index, edge_attr, active_nodes, active_edges, batch = generator(Z, temperature=temperature, use_random=use_random)

            if use_exp:
                #print("nodes_labels.size,edge_index.size,edge_attr.size",nodes_labels.size(),edge_index.size(),edge_attr.size())
                if use_hard:
                    nodes_labels_hard, edge_index_hard, edge_attr_hard, active_nodes_hard, active_edges_hard, batch_hard = temp_hard
                    loss_exp = self.construct_loss(surrogate, nodes_labels_hard, edge_index_hard, edge_attr_hard, active_nodes_hard, active_edges_hard, batch_hard, temperature=tau, use_random=use_random, log_rec_loss=recon_loss)
                else:
                    loss_exp = self.construct_loss(surrogate, nodes_labels, edge_index, edge_attr, active_nodes, active_edges, batch, temperature=tau, use_random=use_random, log_rec_loss=None)

            else:
                loss_exp = torch.tensor([0.0])
            #print("1.5")
            loss_reg_valid = torch.tensor([0.0])
            loss_reg_connect = torch.tensor([0.0])
            loss_reg_diver = torch.tensor([0.0])
            if use_regularize:
                #the strong connectivity constraint
                row, col = edge_index
                a = -100.
                A = torch.sparse.FloatTensor(edge_index,active_edges,torch.Size([batch_size*generator.max_nodes,batch_size*generator.max_nodes])).to_dense()
                #print("A=",A)
                A_list = [torch.eye(batch_size*generator.max_nodes), A]
                #print("A[row, col]=",A[row, col],A[row, col].sum())
                for _ in range(2,generator.max_nodes):
                    A_mm = torch.mm(A_list[-1],A)
                    #print("A_mm[row, col]=",A_mm[row, col],A_mm[row, col].sum())
                    A_temp = torch.zeros_like(A_mm)
                    A_temp[row, col] = 1. / (1. + (a * ( A_mm[row, col] - 0.5)).exp())
                    #print("A_temp=",A_temp)
                    #print("A_temp[row, col]=",A_temp[row, col],A_temp[row, col].sum())
                    A_list.append(A_temp)
                sum_A = sum(A_list)
                C = 1. / (1. + (a * (sum_A[row, col] - 0.5)).exp())
                
                #print("C=",C,C.sum())

                x_idx = torch.arange(batch_size*generator.max_nodes).long()
                input_idx = (torch.arange(0,batch_size*generator.max_nodes,generator.max_nodes).float().view(batch_size,1)*torch.ones(batch_size,generator.max_nodes)).view(-1).long()
                output_idx = (torch.arange(generator.max_nodes-1,batch_size*generator.max_nodes,generator.max_nodes).float().view(batch_size,1)*torch.ones(batch_size,generator.max_nodes)).view(-1).long()

                C_input = 1. / (1. + (a * (sum_A[input_idx, x_idx] - 0.5)).exp())
                C_output = 1. / (1. + (a * (sum_A[x_idx, output_idx] - 0.5)).exp())

                loss_reg_connect_indeg = active_nodes.view(-1) * (1. - C_input) + (1. - active_nodes.view(-1)) * scatter_add(C, col, dim=0, out=torch.zeros(batch_size*generator.max_nodes))
                loss_reg_connect_outdeg = active_nodes.view(-1) * (1. - C_output) + (1. - active_nodes.view(-1)) * scatter_add(C, row, dim=0, out=torch.zeros(batch_size*generator.max_nodes))

                loss_reg_connect = loss_reg_connect_indeg[loss_reg_connect_indeg>SMALL_NUMBER].sum()
                loss_reg_valid = loss_reg_connect_outdeg[loss_reg_connect_outdeg>SMALL_NUMBER].sum()

            total_loss = self.lam_1 * loss_exp + self.lam_2 * loss_reg_valid + self.lam_3 * loss_reg_connect + loss_elbo
            #total_loss =  self.lam_2 * loss_reg_valid + self.lam_3 * loss_reg_connect
            
            total_loss.backward()
            
            nn.utils.clip_grad_norm_(generator.params, 50)
            
            optimizer_gen.step()
            
            loss_store["loss_d"].append(loss_d.item())
            loss_store["loss_g"].append(loss_g.item())
            loss_store["loss_exp"].append(loss_exp.item())
            loss_store["loss_reg_valid"].append(loss_reg_valid.item())
            loss_store["loss_reg_connect"].append(loss_reg_connect.item())
            loss_store["loss_reg_diver"].append(loss_reg_diver.item())
            loss_store["loss_elbo"].append(loss_elbo.item())
            loss_store["total_loss"].append(total_loss.item())
            
            
            #print("loss=",loss.item())

            if True:
                count=0
                for p in generator.params:
                    count+=1
                    #print("%s p.grad"%count,p.grad)
            
            #print("generator.mu_z,generator.log_var_z",generator.mu_z,generator.log_var_z)
            
            
            if loss_opt > total_loss.item():
                generator.save_model("models/opt_score")
                loss_opt = total_loss.item()
                epoch_opt = epoch+1
        
            epoch_time = time.time() - total_time_start
            print("training generator : epoch %s, total loss: %.5f [ loss_d: %.5f, loss_g: %.5f, loss_exp: %.5f, loss_reg_valid: %.5f, loss_reg_connect:  %.5f, loss_reg_diver: %.5f, loss_elbo: %.5f ] | time cost: %.2f" % (epoch+1,total_loss.item(),loss_d.item(),loss_g.item(),loss_exp.item(),loss_reg_valid.item(),loss_reg_connect.item(),loss_reg_diver.item(), loss_elbo.item(), epoch_time),end="\n")
            
            if temperature_annealing_rate is not None:
                tau = max(tau * temperature_annealing_rate,0.01)
        
        
        for p in surrogate.encoder.parameters():
            p.requires_grad=True
        for p in surrogate.predictor.parameters():
            p.requires_grad=True
        
        
        #pl.plot(loss_store["loss_d"],'b')
        #pl.plot(loss_store["loss_g"],'g')
        pl.plot(loss_store["loss_exp"],'r')
        pl.plot(loss_store["loss_reg_valid"],'c')
        pl.plot(loss_store["loss_reg_connect"],'y')
        pl.plot(loss_store["total_loss"],'k')
        pl.legend(["loss_exp",'loss_reg_valid',"loss_reg_connect","total_loss"])
        #pl.show()
        #pl.savefig("gen_score_log.png")
        #pl.close()
        #pl.plot(loss_store["total_loss"],'k')
        #pl.savefig("gen_score_log_total_loss.png")
        #pl.close()
        #pl.plot(loss_store["loss_exp"],'k')
        #pl.savefig("gen_score_log_loss_exp.png")
        #pl.close()
        #pl.plot(loss_store["loss_reg_valid"],'k')
        #pl.savefig("gen_score_log_loss_reg_valid.png")
        #pl.close()
        #pl.plot(loss_store["loss_reg_connect"],'k')
        #pl.savefig("gen_score_log_loss_reg_connect.png")
        #pl.close()
        #pl.plot(loss_store["loss_elbo"],'k')
        #pl.savefig("gen_score_log_loss_elbo.png")
        #pl.close()
        

                                
        #print("\noptimal generator's score = ",loss_opt," at epoch ",epoch_opt)
        #generator.restore_model("models/opt_score")
        
        return total_loss.item()
                    
            
    def weights_init(self, m):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


#------------------------------------------------------------
#Train the Surrogate
#------------------------------------------------------------
class TrainSurrogate():
    def __init__(self, surrogate):
        
        self.surrogate = surrogate
        self.mseloss_fn=MSELoss()
        self.hidden=None
        self.labels=None

    def train(self, data, batch_size=50, learning_rate=1e-4, weight_decay=1e-5, seed=0, num_epochs=2000, convergence_err=1e-3):
        total_time_start = time.time()
        #print("data size=",len(data))
        
        #self.surrogate.encoder.reset_parameters()
        #self.surrogate.predictor.reset_parameters()
        
        optimizer = torch.optim.Adam(list(self.surrogate.encoder.parameters())+list(self.surrogate.predictor.parameters()), lr=learning_rate, weight_decay=weight_decay)
        
        loss_store=[]
        epoch = -1
        avg_loss = 10.
        while avg_loss > convergence_err and (epoch+1) < num_epochs:
            epoch += 1
        #for epoch in range(0, num_epochs):
            
            self.surrogate.encoder.train()
            self.surrogate.predictor.train()
            
            loss = 0.0
            processed_graphs = 0
            
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
            step = 0
            for batch in data_loader:
                step += 1
                
                processed_graphs += batch.num_graphs
                
                optimizer.zero_grad()
                
                #print("batch.batch=",batch.batch)
                #print("batch.edge_attr=",torch.sum(batch.edge_attr,0))
                #print("batch.edge_index=",batch.edge_index)
                
                #mask_edge = (torch.sum(batch.edge_attr!=0,1) > 0)
                encoder_out_mu, _ = self.surrogate.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch, mask_edge = None)
                
                labels_pre = self.surrogate.predictor(encoder_out_mu)
            
                loss_pre = self.mseloss_fn(batch.y[:].view(-1),labels_pre.view(-1))
                
                loss_pre.backward()
                optimizer.step()
                
                loss += loss_pre.item()
                
                avg_loss = loss/processed_graphs
                
                epoch_time = time.time() - total_time_start
                if (epoch+1)%50 == 0 or num_epochs < 50 or step % 50 == 0:
                    print("training surrogate : epoch %s batch %s (has %s graphs), loss: %.5f| time cost: %.2f" % (epoch+1, step, batch.num_graphs, loss/processed_graphs, epoch_time),end="\r")

    
            loss_store.append(loss/processed_graphs)
    
        pl.plot(loss_store,'k')
        pl.legend(["loss"])
        #pl.show()
        #pl.savefig("surrogate_loss_log.png")
        pl.close()
            
        print("Surrogate has been trained.")

    
    def eval_surrogate(self,valid=None,name="test",tau=None):
        #test the surrogate
        if valid is not None:
            self.surrogate.encoder.train()
            self.surrogate.predictor.train()
            data_loader = DataLoader(valid, batch_size=len(valid), shuffle=False)
            for batch in data_loader:
                
                pre_mu, pre_sigma2, rmse_mc, test_ll = self.surrogate.predict(batch.x, batch.edge_index, batch.edge_attr, batch.batch, 10, uncertainty=True, tau=tau, y_test=batch.y[:].view(-1))

                pre_mu_wo, _, rmse, _ = self.surrogate.predict(batch.x, batch.edge_index, batch.edge_attr, batch.batch, None, uncertainty=False, tau=None, y_test=batch.y[:].view(-1))
        
                print("real=",batch.y[:].view(-1))
                print("pre_mu_wo=",pre_mu_wo.view(-1))
                print("pre_mu=",pre_mu.view(-1))
                print("pre_sigma2=",pre_sigma2)
                print("test loss=",rmse)
                print("test loss=",rmse_mc)
        
        
            pl.scatter(torch.linspace(1,len(valid),len(valid)).numpy(),batch.y[:].view(-1).numpy(),c='k',alpha=0.5)
            pl.scatter(torch.linspace(1,len(valid),len(valid)).numpy(),pre_mu_wo.view(-1).detach().numpy(),c='b',alpha=0.5)
            pl.errorbar(torch.linspace(1,len(valid),len(valid)).numpy(),pre_mu.detach().numpy(),yerr=torch.sqrt(pre_sigma2.detach()).numpy(),c='r',alpha=0.5,ecolor='g',capsize=3)
            pl.savefig("eval_surr_%s.png"%name)
            pl.close()
    
            return self.mseloss_fn(batch.y[:].view(-1),pre_mu.view(-1)).item()


#==============================================================================
# Embed Layer and GN Block
#A meta layer for building any kind of graph network, inspired by the
#"Relational Inductive Biases, Deep Learning, and Graph Networks"
# <https://arxiv.org/abs/1806.01261>_ paper
#==============================================================================

class EmbedEdgeModel(torch.nn.Module):
    def __init__(self, input_dim_e, e_mlp, input_dim_n, n_mlp, act, dropout=0.5):
        super(EmbedEdgeModel, self).__init__()
        self.edge_mlp = torch.nn.Sequential()
        input_dim=input_dim_e
        for i in range(len(e_mlp)):
            self.edge_mlp.add_module("Embed_linear%s_in_edge"%(i+1),Lin(input_dim, e_mlp[i]))
            input_dim=e_mlp[i]
            self.edge_mlp.add_module("Embed_relu%s_in_edge"%(i+1),act())
            self.edge_mlp.add_module("Embed_dropout%s_in_edge"%(i+1),torch.nn.Dropout(p=dropout))
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.edge_mlp]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
            else:
                for module in item._modules.values():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
    
    
    def forward(self, edge_attr, batch, mask_edge=None):
        # edge_attr: [E, F_e]
        # batch: [E] with max entry B - 1.
        
        if mask_edge is not None:
            in_filtered = edge_attr[mask_edge]
        else:
            in_filtered = edge_attr
        
        out_filtered = self.edge_mlp(in_filtered) #M*D
        
        if mask_edge is not None:
            out = torch.zeros(edge_attr.size(0),out_filtered.size(1)).to(out_filtered)
            out[mask_edge] = out_filtered
            return out
        else:
            return out_filtered

    def soft_forward(self, edge_attr, batch, weights=None):
        
        out = weights.view(edge_attr.size(0),1) * self.edge_mlp(edge_attr) #M*D
        
        return out

class EmbedNodeModel(torch.nn.Module):
    def __init__(self, input_dim_e, e_mlp, input_dim_n, n_mlp, act, dropout=0.5):
        super(EmbedNodeModel, self).__init__()
        self.node_mlp = torch.nn.Sequential()
        input_dim=input_dim_n
        for i in range(len(n_mlp)):
            self.node_mlp.add_module("Embed_linear%s_in_node"%(i+1),Lin(input_dim, n_mlp[i]))
            input_dim=n_mlp[i]
            self.node_mlp.add_module("Embed_relu%s_in_node"%(i+1),act())
            self.node_mlp.add_module("Embed_dropout%s_in_node"%(i+1),torch.nn.Dropout(p=dropout))
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_mlp]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
            else:
                for module in item._modules.values():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
    
    def forward(self, x, batch):
        # x: [N, F_x], where N is the number of nodes.
        # batch: [N] with max entry B - 1.
        return self.node_mlp(x) #N*D

    def soft_forward(self,  x, batch, weights=None):
        
        return weights.view(x.size(0),1) * self.node_mlp(x) #N*D

class EmbedMetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super(EmbedMetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
            else:
                for module in item._modules.values():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, mask_edge=None):
        row, col = edge_index
        
        if self.edge_model is not None:
            edge_attr = self.edge_model(edge_attr,
                                        batch if batch is None else batch[row], mask_edge)
    
        if self.node_model is not None:
            x = self.node_model(x, batch)
            
        return x, edge_attr
            
    def soft_forward(self, x, edge_index, edge_attr=None, batch=None, weights_nodes=None, weights_edges=None):
        row, col = edge_index

        if self.edge_model is not None:
            edge_attr = self.edge_model.soft_forward(edge_attr,
                                        batch if batch is None else batch[row], weights_edges)

        if self.node_model is not None:
            x = self.node_model.soft_forward(x, batch, weights_nodes)
    
        return x, edge_attr
    
    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model)


class EdgeModel(torch.nn.Module):
    def __init__(self, input_dim_e, e_mlp, input_dim_n, n_mlp, act, dropout=0.5):
        super(EdgeModel, self).__init__()
        self.edge_mlp = torch.nn.Sequential()
        input_dim=input_dim_n+input_dim_n+input_dim_e
        for i in range(len(e_mlp)):
            self.edge_mlp.add_module("GNN_linear%s_in_edge"%(i+1),Lin(input_dim, e_mlp[i]))
            input_dim=e_mlp[i]
            self.edge_mlp.add_module("GNN_relu%s_in_edge"%(i+1),act())
            self.edge_mlp.add_module("GNN_dropout%s_in_edge"%(i+1),torch.nn.Dropout(p=dropout))
        self.reset_parameters()
                
    def reset_parameters(self):
        for item in [self.edge_mlp]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
            else:
                for module in item._modules.values():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

    def forward(self, src, dest, edge_attr, batch, mask_edge=None):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # batch: [E] with max entry B - 1.
        # mask_edge : [E] is used to mask the edges with the original features [0,0,0...]
        input = torch.cat([src, dest, edge_attr], 1)
        
        if mask_edge is not None:
            in_filtered = input[mask_edge]
        else:
            in_filtered = input
        
        out_filtered = self.edge_mlp(in_filtered) #M*D
        
        if mask_edge is not None:
            out = torch.zeros(edge_attr.size(0),out_filtered.size(1)).to(out_filtered)
            out[mask_edge] = out_filtered
            return out
            #edge_attr[mask_edge] = out_filtered
            #return edge_attr
        else:
            return out_filtered

    def soft_forward(self, src, dest, edge_attr, batch, weights=None):
        #print("src.size(),dest.size(),edge_attr.size()",src.size(),dest.size(),edge_attr.size())
        input = torch.cat([src, dest, edge_attr], 1)
        
        out = weights.view(input.size(0),1) * self.edge_mlp(input) #M*D
        
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, input_dim_e, e_mlp, input_dim_n, n_mlp, act, dropout=0.5):
        super(NodeModel, self).__init__()
        self.node_mlp = torch.nn.Sequential()
        input_dim=input_dim_n+e_mlp[-1]
        for i in range(len(n_mlp)):
            self.node_mlp.add_module("GNN_linear%s_in_node"%(i+1),Lin(input_dim, n_mlp[i]))
            input_dim=n_mlp[i]
            self.node_mlp.add_module("GNN_relu%s_in_node"%(i+1),act())
            self.node_mlp.add_module("GNN_dropout%s_in_node"%(i+1),torch.nn.Dropout(p=dropout))
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_mlp]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
            else:
                for module in item._modules.values():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = scatter_add(edge_attr, col, dim=0, out=torch.zeros(x.size(0),edge_attr.size(1)))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp(out) #N*D
    
    def soft_forward(self, x, edge_index, edge_attr, batch, weights=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = scatter_add(edge_attr, col, dim=0, out=torch.zeros(x.size(0),edge_attr.size(1)))
        out = torch.cat([x, out], dim=1)
        return weights.view(out.size(0),1) * self.node_mlp(out) #N*D

class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
            else:
                for module in item._modules.values():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, mask_edge=None):
        """"""
        row, col = edge_index
        
        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr,
                                        batch if batch is None else batch[row], mask_edge)
        
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, batch)
    
        return x, edge_attr
            
    def soft_forward(self, x, edge_index, edge_attr=None, batch=None, weights_nodes=None, weights_edges=None):
        """"""
        row, col = edge_index
        
        if self.edge_model is not None:
            edge_attr = self.edge_model.soft_forward(x[row], x[col], edge_attr,
                                        batch if batch is None else batch[row], weights_edges)
        
        if self.node_model is not None:
            x = self.node_model.soft_forward(x, edge_index, edge_attr, batch, weights_nodes)
        
        return x, edge_attr

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model)

#==============================================================================

#------------------------------------------------------------
#GenCandidates is used to generate some candidates
#------------------------------------------------------------
class GenCandidates():
    def __init__(self, dataset='qm9'):
        self.dataset=dataset
        self.observed_data = {"mol":[],"y":[]}
        self.use_prob_temp=use_prob_temp
    
    def generate_candidates_randomly(self, data=None, filter_flag=True, num_gen=100):
        print("generating the candidates randomly or randomly choosing from existing data")
        if data is not None:
            print("generate_candidates_randomly")
            exit(1)
    
    def generate_candidates(self, generator, Z=None, key=None, filter_flag=True, num_gen=100, use_random=False):
        
        print("generating the candidates")
        
        generator.eval()
                        
        if Z is None:
            Z = torch.randn(num_gen,generator.input_dim)
        if self.use_prob_temp:
            graphs = generator.hard_decode_prob_temp(Z, use_random=use_random, decode_times=6)[0]
        else:
            graphs = generator.hard_decode(Z, use_random=use_random)

        #Converting the graphs to sparse
        graphs_sparse = []
        count = 0
        for x, edge_index, edge_attr, active_nodes, active_edges, log_prob in graphs:
            count += 1
            #print("in=",[x, edge_index, edge_attr, active_nodes, active_edges])
            
            x_sparse = x[active_nodes>0].view(-1,generator.num_node_type)
            #print(x[-1,:],active_nodes)
            # print(edge_index.size(1),edge_attr.size(0))
            
            edge_index_sparse_old = edge_index[:,active_edges>0]
            edge_attr_sparse_old = edge_attr[active_edges>0]        

            #print("edge_index_sparse_old=",edge_index_sparse_old)
            
            #node hash
            nodes_idx = torch.nonzero(active_nodes>0).view(-1)
            #print("nodes_idx=",nodes_idx)
            
            #remove the ghost edges
            edge_index_sparse_filtered = []
            edge_attr_sparse_filtered = []
            for idx in range(edge_index_sparse_old.size(1)):
                if edge_index_sparse_old[0,idx] in nodes_idx and edge_index_sparse_old[1,idx] in nodes_idx:
                    edge_index_sparse_filtered.append([edge_index_sparse_old[0,idx],edge_index_sparse_old[1,idx]])
                    edge_attr_sparse_filtered.append(edge_attr_sparse_old[idx].view(1,-1))
            # print(len(edge_index_sparse_filtered),len(edge_attr_sparse_filtered))
            edge_index_sparse_filtered = torch.tensor(edge_index_sparse_filtered).t().long()
            #print("edge_index_sparse_filtered=",edge_index_sparse_filtered)

            nodes_idx_filtered=[]
            if len(edge_index_sparse_filtered)>0:
                #remove the ghost nodes
                for idx in nodes_idx:
                    if idx in edge_index_sparse_filtered[0] or idx in edge_index_sparse_filtered[1]:
                        nodes_idx_filtered.append(idx)
                #print(nodes_idx_filtered)
                x_sparse = x[nodes_idx_filtered,:]
            
            
            #hash
            edge_index_sparse = torch.zeros_like(edge_index_sparse_filtered)
            
            #print("edge_index_sparse_filtered=",edge_index_sparse_filtered)
            for idx in range(len(nodes_idx_filtered)):
                edge_index_sparse[edge_index_sparse_filtered==nodes_idx_filtered[idx]]=idx
        
            if len(edge_attr_sparse_filtered) > 0:
                edge_attr_sparse = torch.cat(edge_attr_sparse_filtered,dim=0)
            else:
                edge_attr_sparse = torch.tensor([]).view(0,edge_attr.size(1))
                edge_index_sparse = torch.tensor([]).view(2,0)
            
            #print("edge_index_sparse=",edge_index_sparse)
            #print(edge_attr_sparse.size(0),edge_index_sparse.size(1))
            #assert edge_attr_sparse.size(0)==edge_index_sparse.size(1)
                        
            n=x_sparse.size(0)
            MAX_NODES=x.size(0)
            if n < MAX_NODES:
                #padding zeros into x to ensure the first node is input and the 20th one is output
                x_sparse=torch.cat([x_sparse[0:n-1,:],torch.zeros(MAX_NODES-n,x_sparse.size(1)),x_sparse[n-1,:].view(1,-1)],dim=0)
                edge_index_sparse[edge_index_sparse==n-1]=MAX_NODES-1
            
            graphs_sparse.append([x_sparse,edge_index_sparse.long(),edge_attr_sparse,log_prob])
            print("%s/%s"%(count,Z.size(0)),end='\r')
    
            #print("graph=",[x_sparse,edge_index_sparse,edge_attr_sparse],edge_index_sparse_old,nodes_idx,active_edges)
            #exit(1)
        
        generator.train()
        
        if filter_flag:
            #return self.is_valid(graphs_sparse)
            return self.is_valid_NASBench201(graphs_sparse)
        else:
            return graphs_sparse
    
    def init_observed_data(self, dataset):
        #init mols with observed_graphs
        for d in dataset:
            """
            mol = to_mol({'x':d.x,'edge_index':d.edge_index,'edge_attr':d.edge_attr},dataset=self.dataset)
            if mol is None:
                new_mol = ""
            else:
                new_mol=Chem.MolToSmiles(mol)
            self.observed_data["mol"].append(new_mol)
            """
            self.observed_data["y"].append(d.y[0])

    def update_observed_data(self,new_graphs,ys):
        #init mols with observed_graphs
        count=-1
        for g in new_graphs:
            count+=1
            """
            mol = to_mol({'x':g[0],'edge_index':g[1],'edge_attr':g[2]},dataset=self.dataset)
            #mol = to_mol({'x':d.x,'edge_index':d.edge_index,'edge_attr':d.edge_attr})
            if mol is None:
                new_mol = ""
            else:
                new_mol=Chem.MolToSmiles(mol)
            self.observed_data["mol"].append(new_mol)
            """
            self.observed_data["y"].append(ys[count]["y"])
    
    def is_valid(self,graphs):
        filtered_graphs = []
        for g in graphs:
            #print("haha",g)
            x_sparse,edge_index_sparse,edge_attr_sparse,_ = g

            if x_sparse.size(0) == 0:
                print("non-valid due to [x_sparse.size(0) == 0]")
                continue
            #the first is input and last is output and #only one input and only one output
            if x_sparse[0][0] == 1 and x_sparse[x_sparse.size(0)-1][1] == 1 and sum(x_sparse[:,0])==1 and sum(x_sparse[:,1])==1:
                valid_nodes_from_edges = list(set(list(edge_index_sparse[0].numpy())+list(edge_index_sparse[1].numpy())))      
                print("nodes=",valid_nodes_from_edges)
                DG = nx.DiGraph()
                nodes=[int(n) for n in valid_nodes_from_edges]
                flag_nodes=True
                for n in nodes:
                    if len(x_sparse[n].nonzero().view(-1))==0:
                        print("non-valid due to [len(x_sparse[n].nonzero().view(-1))==0]")
                        flag_nodes=False
                        break
                if flag_nodes==False or 0 not in nodes or int(x_sparse.size(0)-1) not in nodes:
                    print("non-valid due to [flag_nodes==False]")
                    continue
                #print(n,"add",int(x_sparse[n].nonzero().view(-1)[0]))
                #DG[n]['x_label']=int(x_sparse[n].nonzero().view(-1)[0])
                    

                DG.add_edges_from([(int(edge[0]),int(edge[1])) for edge in edge_index_sparse.t().numpy()])
                print("edges=",DG.edges())
                res_con_first=[]
                res_con_last=[]
                for node in nodes:
                    #node 0 connect to any one
                    if node != 0:
                        res_con_first.append(nx.node_connectivity(DG, 0, node) ) 
                    #any one connect to node n-1
                    if node != int(x_sparse.size(0)-1):
                        res_con_last.append(nx.node_connectivity(DG, node, int(x_sparse.size(0)-1)))

                print("connection check: 0->x:",res_con_first,"x->n-1:",res_con_last)
                if 0 not in res_con_first and 0 not in res_con_last:
                    print("valid +1")
                    filtered_graphs.append(g)
                else:
                    print("non-valid due to [0 in res_con_first or 0 in res_con_first]")

        print("valid size=",len(filtered_graphs),"/",len(graphs))
        return filtered_graphs

    def is_valid_NASBench201(self,graphs):
        return graphs

    def filter(self, graphs, key=None):
        mols = []
        #filtering
        filtered_graphs = []
        for g in graphs:
            
            if g[0].size(0) == 0:
                continue
            
            mol = to_mol({'x':g[0],'edge_index':g[1],'edge_attr':g[2]},dataset=self.dataset)
            
            if mol is None:
                new_mol = ""
            else:
                new_mol=Chem.MolToSmiles(mol)
            
            print("new_mol=",new_mol)
            
            if mol is not None and new_mol not in mols and new_mol not in self.observed_data["mol"]:
                mols.append(new_mol)
                if key is None:
                    filtered_graphs.append(g)
                else:
                    filtered_graphs.append({new_mol:g})
        return filtered_graphs

def augment_into(dataset, g, comments=None):
    
    d = Data(x=g['x'],edge_index=g['edge_index'],edge_attr=g['edge_attr'],y=g['y'])

    dataset.add(d)
    
    if comments is not None:
        print("augmented one \"%s\". data size = %s"%(comments,len(dataset)))
    
#------------------------------------------------------------
#Some useful classes related to the multiprocessing
#------------------------------------------------------------

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

#------------------------------------------------------------
#Choose next graph to evaluate
#------------------------------------------------------------
class ChooseNext():
    def __init__(self, surrogate, constraint=None, generator=None, vaeencoder=None):
        self.surrogate = surrogate
        self.constraint = constraint
        self.generator = generator
        self.vaeencoder = vaeencoder
        self.dataset=None
        self.trainGen=None
        self.opt_s=None
        self.maxy=None
        self.z_observed=None
        self.y_observed=None
    
    def next(self, candidates, acq_type):
        best_one_idx = self.do_search_speedup(candidates, acq_type=acq_type, walkers=8)
        return [candidates[best_one_idx]],best_one_idx

    def score_one_can_Rand(self, d):
        score = self.acq_Rand(None,None).view(-1).detach().numpy()[0]
        #print(score)
        return score

    def score_one_can_EI(self, d):
        #print(d)
        #print("(1) start in score_one_can",d)
        #print(self.surrogate.predictor.observed_embeddings.detach(),self.surrogate.predictor.observed_embeddings.size(), self.surrogate.predictor.observed_y.size())
        #zstar_, _ = self.surrogate.encoder(d[0], d[1], d[2], torch.tensor([ 0.] * len(d[0])).to(d[0]).long())

        pre_mu, pre_sigma2, _, _ = self.surrogate.predict(d[0], d[1], d[2], torch.tensor([ 0.] * len(d[0])).to(d[0]).long(), weights_nodes=None, weights_edges=None, NSample=10, uncertainty=True, soft=False, tau=None, y_test=None)

        #print("(2) start in score_one_can")
        score = self.acq_EI(pre_mu, torch.sqrt(pre_sigma2)).view(-1).detach().numpy()[0]
        #print(score)
        return score

    def score_one_can_Std(self, d):
        #print(d)
        pre_mu, pre_sigma2, _, _ = self.surrogate.predict(d[0], d[1], d[2], torch.tensor([ 0.] * len(d[0])).to(d[0]).long(), weights_nodes=None, weights_edges=None, NSample=10, uncertainty=True, soft=False, tau=None, y_test=None)

        #print("(2) start in score_one_can")
        score = self.acq_Std(pre_mu, torch.sqrt(pre_sigma2)).view(-1).detach().numpy()[0]
        #print(score)
        return score
    
    def do_search_speedup(self, candidates, acq_type="EI", walkers=1):
        #Exhausting candidate set
        start_time=time.time()
        
        #print("start")
        self.surrogate.encoder.eval()
        self.surrogate.predictor.eval()
        
        #pre-compute the K_inv and m using the observations
        #self.z_observed = self.surrogate.predictor.linear_predictor(self.surrogate.predictor.observed_embeddings).detach()
        #self.y_observed = self.surrogate.predictor.observed_y.detach()
        
        with MyPool(walkers) as p:
            #for d in candidates:
            #    p.apply_async(self.score_one_can,(d,))
            if acq_type == "EI":
                self.maxy = max(self.y_observed)
                res = p.map(self.score_one_can_EI,candidates)
            elif acq_type == "Std":
                #self.maxy = max(self.surrogate.predictor.observed_y.view(-1))
                res = p.map(self.score_one_can_Std,candidates)
            elif acq_type == "Rand":
                res = p.map(self.score_one_can_Rand,candidates)
            else:
                print("error: acq_type %s is not identified. {EI, EGI, Rand} are implemented. "%acq_type)
                exit(1)
            
            p.close()
            p.join()

        best_one_idx = np.argmax(res)
        print("acq_values=",res,"max idx=",best_one_idx,"max value=",res[best_one_idx],"cost=",time.time()-start_time)
        return best_one_idx
            
    #-------Acquisition Functions----------

    #Acquisition Function : EGI
    def acq_Rand(self,mu,std, weights=None):
        return torch.randn(1)
    
    #Acquisition Function : EI
    def acq_EI(self, mu, std, weights=None):
        maxy = self.maxy
        gamma = (mu-maxy)/std
        
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        #print("m in EI",m.log_prob(gamma).exp(),"gamma=",gamma,"m.cdf(gamma)=",m.cdf(gamma),"(mu-maxy)/std",mu,maxy,std)
        pdfgamma=m.log_prob(gamma).exp()
        cdfgamma=m.cdf(gamma)
        result=std*(pdfgamma+gamma*cdfgamma)
        
        return result

    #Acquisition Function : EGI
    def acq_EGI(self,mu,std, weights=None):
        maxy = self.maxy
        gamma = (mu-maxy)/std
        
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        #print("m in EI",m.log_prob(gamma).exp(),"gamma=",gamma,"m.cdf(gamma)=",m.cdf(gamma),"(mu-maxy)/std",mu,maxy,std)
        pdfgamma=m.log_prob(gamma).exp()
        cdfgamma=m.cdf(gamma)
        result=std*(pdfgamma+gamma*cdfgamma)
        
        return result

    #Acquisition Function : weightEI
    def acq_weightEI(self, mu, std, weights=None):
        maxy = self.maxy
        gamma = (mu-maxy)/std
        
        m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        #print("m in EI",m.log_prob(gamma).exp(),"gamma=",gamma,"m.cdf(gamma)=",m.cdf(gamma),"(mu-maxy)/std",mu,maxy,std)
        pdfgamma=m.log_prob(gamma).exp()
        cdfgamma=m.cdf(gamma)
        result=std*(pdfgamma+gamma*cdfgamma)
        
        return result*weights

    #Acquisition Function : UCB
    def acq_UCB(self, mu, std, weights=None):
        return mu + eta * std
    
    #Acquisition Function : Std
    def acq_Std(self, mu, std, weights=None):
        return std



#------------------------------------------------------------
#to test the Generative Quality
#------------------------------------------------------------
class testGenerativeQuality():
    def __init__(self, filename="results/observations_mcdropout.csv", target='joint', dataset='qm9'):
        self.target = target
        self.dataset = dataset
        self.filename = filename
        self.count = 0
        #self.min = -10.0
        #self.max = 2.0
        self.min = 0.0
        self.max = 1.0
        self.start_time = time.time()

    def evaluate_NN(self,input_g):
        res = objective_func.evaluate_point(input_g)
        return res
    
    #convert graph to string representation in NASBench201
    def get_archstr(self,input_g):
        x = input_g[0]
        edge_index = input_g[1]
        edge_attr = input_g[2]
        
        def idx2opsstr(idx):
            ops_str_list = ["none","skip_connect","nor_conv_1x1","nor_conv_3x3","avg_pool_3x3"]
            return ops_str_list[idx]
        
        op_str_1_0 = idx2opsstr(torch.nonzero(edge_attr[0]).item()) #0->1
        op_str_2_0 = idx2opsstr(torch.nonzero(edge_attr[1]).item()) #0->2
        op_str_2_1 = idx2opsstr(torch.nonzero(edge_attr[3]).item()) #1->2
        op_str_3_0 = idx2opsstr(torch.nonzero(edge_attr[2]).item()) #0->3
        op_str_3_1 = idx2opsstr(torch.nonzero(edge_attr[4]).item()) #1->3
        op_str_3_2 = idx2opsstr(torch.nonzero(edge_attr[5]).item()) #2->3
        
        arch_str = '|%s~0|+|%s~0|%s~1|+|%s~0|%s~1|%s~2|'%(op_str_1_0,op_str_2_0,op_str_2_1,op_str_3_0,op_str_3_1,op_str_3_2)
    
        return arch_str

    # This function is to mimic the training and evaluatinig procedure for a single architecture `arch`.
    # The time_cost is calculated as the total training time for a few (e.g., 12 epochs) plus the evaluation time for one epoch.
    # For use_012_epoch_training = True, the architecture is trained for 12 epochs, with LR being decaded from 0.1 to 0.
    #       In this case, the LR schedular is converged.
    # For use_012_epoch_training = False, the architecture is planed to be trained for 200 epochs, but we early stop its procedure.
    #       
    def train_and_eval(self, arch, nas_bench, extra_info, dataname='cifar10-valid', use_012_epoch_training=True):

        if use_012_epoch_training and nas_bench is not None:
            arch_index = nas_bench.query_index_by_arch( arch )
            assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
            info = nas_bench.get_more_info(arch_index, dataname, None, True)
            valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
            #_, valid_acc = info.get_metrics('cifar10-valid', 'x-valid' , 25, True) # use the validation accuracy after 25 training epochs
        elif not use_012_epoch_training and nas_bench is not None:
            # Please contact me if you want to use the following logic, because it has some potential issues.
            # Please use `use_012_epoch_training=False` for cifar10 only.
            # It did return values for cifar100 and ImageNet16-120, but it has some potential issues. (Please email me for more details)
            arch_index, nepoch = nas_bench.query_index_by_arch( arch ), 25
            assert arch_index >= 0, 'can not find this arch : {:}'.format(arch)
            xoinfo = nas_bench.get_more_info(arch_index, 'cifar10-valid', None, True)
            xocost = nas_bench.get_cost_info(arch_index, 'cifar10-valid', False)
            info = nas_bench.get_more_info(arch_index, dataname, nepoch, False, True) # use the validation accuracy after 25 training epochs, which is used in our ICLR submission (not the camera ready).
            cost = nas_bench.get_cost_info(arch_index, dataname, False)
            # The following codes are used to estimate the time cost.
            # When we build NAS-Bench-201, architectures are trained on different machines and we can not use that time record.
            # When we create checkpoints for converged_LR, we run all experiments on 1080Ti, and thus the time for each architecture can be fairly compared.
            nums = {'ImageNet16-120-train': 151700, 'ImageNet16-120-valid': 3000,
                    'cifar10-valid-train' : 25000,  'cifar10-valid-valid' : 25000,
                    'cifar100-train'      : 50000,  'cifar100-valid'      : 5000}
            estimated_train_cost = xoinfo['train-per-time'] / nums['cifar10-valid-train'] * nums['{:}-train'.format(dataname)] / xocost['latency'] * cost['latency'] * nepoch
            estimated_valid_cost = xoinfo['valid-per-time'] / nums['cifar10-valid-valid'] * nums['{:}-valid'.format(dataname)] / xocost['latency'] * cost['latency']
            try:
                valid_acc, time_cost = info['valid-accuracy'], estimated_train_cost + estimated_valid_cost
            except:
                valid_acc, time_cost = info['valtest-accuracy'], estimated_train_cost + estimated_valid_cost
        else:
            # train a model from scratch.
            raise ValueError('NOT IMPLEMENT YET')
        return valid_acc, time_cost
    
    def evaluate_NASBench201(self,input_g):
        #res = objective_func.evaluate_point(input_g)
        #convert graph to string representation in NASBench201
        arch_str = self.get_archstr(input_g)
        print(">>>>>>>> eval archstr:\"%s\" ..."%arch_str)

        index = api.query_index_by_arch(arch_str)
        
        print(">>>>>>>> hash in NASBench201, idx = %s"%index)
        
        use_12epoch=False
        if use_12epoch :
            valid_acc, time_cost = self.train_and_eval(arch_str, api, None, dataname=image_data, use_012_epoch_training=True)
            res = -1.0*(1.-valid_acc/100.)
            
            #get test results at 12th epoch (not use in search)
            if image_data == "cifar10-valid":
                results_cifar10_12 = api.query_by_index(index, "cifar10", use_12epochs_result=True) # a dict of all trials, where the key is the seed
                results_cifar100_12 = api.query_by_index(index, "cifar100", use_12epochs_result=True)
                results_ImageNet16_120_12 = api.query_by_index(index, "ImageNet16-120", use_12epochs_result=True)
            else:
                #results_12 = api.query_by_index(index, image_data, use_12epochs_result=True) # a dict of all trials, where the key is the seed
                print("eval [%s] data is still not implemented"%image_data)
                exit(1)
            
            #get test results at 200th epoch (not use in search)
            if image_data == "cifar10-valid":
                results_cifar10_200 = api.query_by_index(index, "cifar10", use_12epochs_result=False) # a dict of all trials, where the key is the seed
                results_cifar100_200 = api.query_by_index(index, "cifar100", use_12epochs_result=False) # a dict of all trials, where the key is the seed
                results_ImageNet16_120_200 = api.query_by_index(index, "ImageNet16-120", use_12epochs_result=False) # a dict of all trials, where the key is the seed
            else:
                #results_200 = api.query_by_index(index, image_data, use_12epochs_result=False) # a dict of all trials, where the key is the seed
                print("eval [%s] data is still not implemented"%image_data)
                exit(1)
            
            test_acc_list_cifar10_12 = [results_cifar10_12[trial_seed_i].get_eval("ori-test")["accuracy"] for trial_seed_i in list(results_cifar10_12.keys())]
            test_acc_list_cifar100_12 = [results_cifar100_12[trial_seed_i].get_eval("ori-test")["accuracy"] for trial_seed_i in list(results_cifar100_12.keys())]
            test_acc_list_ImageNet16_120_12 = [results_ImageNet16_120_12[trial_seed_i].get_eval("ori-test")["accuracy"] for trial_seed_i in list(results_ImageNet16_120_12.keys())]
            
            test_acc_list_cifar10_200 = [results_cifar10_200[trial_seed_i].get_eval("ori-test")["accuracy"] for trial_seed_i in list(results_cifar10_200.keys())]
            test_acc_list_cifar100_200 = [results_cifar100_200[trial_seed_i].get_eval("ori-test")["accuracy"] for trial_seed_i in list(results_cifar100_200.keys())]
            test_acc_list_ImageNet16_120_200 = [results_ImageNet16_120_200[trial_seed_i].get_eval("ori-test")["accuracy"] for trial_seed_i in list(results_ImageNet16_120_200.keys())]
            
            #get valid results at 200th epoch (not use in search)
            results_cifar10_valid_200 = api.query_by_index(index, image_data) # a dict of all trials, where the key is the seed
            cifar10_valid_accu_list_200 = [results_cifar10_valid_200[trial_seed_i].get_eval("x-valid")["accuracy"] for trial_seed_i in list(results_cifar10_valid_200.keys())]
            results_cifar100_valid_200 = api.query_by_index(index, "cifar100") # a dict of all trials, where the key is the seed
            cifar100_valid_accu_list_200 = [results_cifar100_valid_200[trial_seed_i].get_eval("x-valid")["accuracy"] for trial_seed_i in list(results_cifar100_valid_200.keys())]
            results_ImageNet16_120_valid_200 = api.query_by_index(index, "ImageNet16-120") # a dict of all trials, where the key is the seed
            ImageNet16_120_valid_accu_list_200 = [results_ImageNet16_120_valid_200[trial_seed_i].get_eval("x-valid")["accuracy"] for trial_seed_i in list(results_ImageNet16_120_valid_200.keys())]
            
            acc_info = {"cifar10":{"valid-200":cifar10_valid_accu_list_200,"test_12":test_acc_list_cifar10_12,"test_200":test_acc_list_cifar10_200},"cifar100":{"valid-200":cifar100_valid_accu_list_200,"test_12":test_acc_list_cifar100_12,"test_200":test_acc_list_cifar100_200},"ImageNet16-120":{"valid-200":ImageNet16_120_valid_accu_list_200,"test_12":test_acc_list_ImageNet16_120_12,"test_200":test_acc_list_ImageNet16_120_200}}
            
            print(">>>>>>>> get result in NASBench201, valid_error = %s (use in search), cost = %s, and acc_info = %s (not use in search, just test)"%(-res,time_cost,acc_info))
        else:

            """
            Args [dataset] (4 possible options):
            -- cifar10-valid : training the model on the CIFAR-10 training set.
            -- cifar10 : training the model on the CIFAR-10 training + validation set.
            -- cifar100 : training the model on the CIFAR-100 training set.
            -- ImageNet16-120 : training the model on the ImageNet16-120 training set.
            Args [setname] (each dataset has different setnames):
            -- When dataset = cifar10-valid, you can use 'train', 'x-valid', 'ori-test'
            ------ 'train' : the metric on the training set.
            ------ 'x-valid' : the metric on the validation set.
            ------ 'ori-test' : the metric on the test set.
            -- When dataset = cifar10, you can use 'train', 'ori-test'.
            ------ 'train' : the metric on the training + validation set.
            ------ 'ori-test' : the metric on the test set.
            -- When dataset = cifar100 or ImageNet16-120, you can use 'train', 'ori-test', 'x-valid', 'x-test'
            ------ 'train' : the metric on the training set.
            ------ 'x-valid' : the metric on the validation set.
            ------ 'x-test' : the metric on the test set.
            ------ 'ori-test' : the metric on the validation + test set.
            """
            results = api.query_by_index(index, image_data) # a dict of all trials, where the key is the seed
            print ('>>>>>>>> there are {:} trials for this architecture on {:}'.format(len(results), image_data))
            
            valid_error_list = [1. - results[trial_seed_i].get_eval("x-valid")["accuracy"]/100. for trial_seed_i in list(results.keys())]
        
            #randomly choose one from multipy trials
            res = -1.0*random.choice(valid_error_list)
        
            time_cost = 0.
        
            if image_data == "cifar10-valid":
                results = api.query_by_index(index, "cifar10") # a dict of all trials, where the key is the seed

            test_acc_list = [results[trial_seed_i].get_eval("ori-test")["accuracy"] for trial_seed_i in list(results.keys())]
            
            acc_info = test_acc_list
            
            print(">>>>>>>> get result in NASBench201, valid_error = %s (use in search), cost = %s, and acc_info = %s (not use in search, just test)"%(-res,time_cost,acc_info))
        
        return res, time_cost, arch_str, index, acc_info
    
    
    #recall the real evaluation function to evaluate
    def evaluate_point(self, input):
        if input["smiles"]=="":
            y_normized = -10.
        else:
            y = objective_func.evaluate_point(input,target=self.target)
            #y_normized = y
            #y_normized = (y-self.min)/(self.max-self.min)
            y_normized = np.exp(y)
        return y_normized

    def denormalize(self,res_list):
        #res_list = [y_normalized * (self.max-self.min) + self.min for y_normalized in res_list]
        res_list = [np.log(y_normalized) for y_normalized in res_list]
        return res_list
        
    def test(self, generated_graphs,store_flag=True):
        mols = self.to_mols(generated_graphs)
        goal_list = []
        for mol in mols:
            goal = self.evaluate_point(mol)
            goal_list.append({"smiles":mol["smiles"],"y":goal,"num_nodes":Chem.MolFromSmiles(mol["smiles"]).GetNumAtoms() if mol["smiles"] is not "" else -1})
            if store_flag:
                self.count += 1
                #save into csv file
                with open(self.filename,"a",newline="") as datacsv:
                    csvwriter = csv.writer(datacsv)
                    if self.count == 1:
                        fieldnames = ["# eval", 'smiles', 'properties', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))]
                        csvwriter.writerow(fieldnames)
                    csvwriter.writerow([self.count, mol["smiles"], goal])
        return goal_list
    
    def test_NN(self, generated_graphs,store_flag=True):
        goal_list = []
        for mol in generated_graphs:
            goal = self.evaluate_NN(mol)
            goal_list.append({"y":goal})
            if store_flag:
                self.count += 1
                #save into csv file
                with open(self.filename,"a",newline="") as datacsv:
                    csvwriter = csv.writer(datacsv)
                    if self.count == 1:
                        fieldnames = ["# eval", 'time', 'properties', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))]
                        csvwriter.writerow(fieldnames)
                    csvwriter.writerow([self.count, time.time()-self.start_time,goal])
                with open('results/architecture_%s_%s_%s.pkl'%(model_name,self.count,int(opts.random_seed)), 'wb') as f:
                    pickle.dump(mol, f)
        return goal_list
    
    def test_NASBench201(self, generated_graphs,store_flag=True):
        goal_list = []
        for mol in generated_graphs:
            goal, time_cost, arch_str, arch_index, test_acc_list = self.evaluate_NASBench201(mol)
            goal_list.append({"y":goal})
            if store_flag:
                self.count += 1
                #save into csv file
                with open(self.filename,"a",newline="") as datacsv:
                    csvwriter = csv.writer(datacsv)
                    if self.count == 1:
                        fieldnames = ["# eval", 'total_search_cost', 'one_eval_cost', 'properties', 'arch_str', 'arch_index','test_acc_list', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))]
                        csvwriter.writerow(fieldnames)
                    csvwriter.writerow([self.count, time.time()-self.start_time, time_cost,goal,arch_str,arch_index,"%s"%test_acc_list])
                #with open('results/architecture_%s_%s_%s.pkl'%(model_name,self.count,int(opts.random_seed)), 'wb') as f:
                #    pickle.dump(mol, f)
        return goal_list
    
    def test_dataset(self,dataset):
        res = []
        for i in range(len(dataset)):
            res.append(dataset[i].y[:].item())
        return res
    
    def draw_dist(self, res_list, ax, color, label):
        sns.distplot(res_list, hist=True, rug=False, ax=ax, color=color,label=label)
    

    def to_mols(self, graphs):
        mols = []
        for g in graphs:
            mol = to_mol({'x':g[0],'edge_index':g[1],'edge_attr':g[2]},dataset=self.dataset)
            
            if mol is None:
                new_mol = ""
            else:
                new_mol=Chem.MolToSmiles(mol)
            
            mols.append({"smiles":new_mol})
        
        return mols

    def metric_valid(self, results_list):
        res_list_filtered = []
        for re in results_list:
            if re['y'] != -10:
                res_list_filtered.append(re)
        return float(len(res_list_filtered))/len(results_list)

    def metric_unique(self, results_list):
        res_list_filtered = []
        for re in results_list:
            if re['y'] != -10:
                res_list_filtered.append(re)
        return float(len(set([res['smiles'] for res in res_list_filtered])))/len(res_list_filtered)

    def metric_novel(self, results_list, observed_data):
        res_list_filtered = []
        for re in results_list:
            if re['y'] != -10:
                res_list_filtered.append(re)
        count = 0.
        for re in res_list_filtered:
            if re["smiles"] not in observed_data["mol"]:
                count += 1.
        return float(count)/len(res_list_filtered)


def draw_dis(generator,Z, count):
    can_test = genCan.generate_candidates(generator, Z=Z, filter_flag=False, num_gen=1000, use_random=use_random)
    results_list = eval_mol.test_NN(can_test,store_flag=False)
    #print(results_list)
    res_list_filtered = []
    for re in results_list:
        if re['y'] != -10:
            re['y'] = eval_mol.denormalize([re['y']])[0]
            res_list_filtered.append(re)
    with open("scores_%s_%s_%s.txt"%(model_name,acq_type,seed),'a') as f:
        f.write("%s\t%s\t%s\t%s\t%s\t%s\n"%(count,np.mean([res['y'] for res in res_list_filtered]),np.std([res['y'] for res in res_list_filtered]),eval_mol.metric_valid(results_list),eval_mol.metric_unique(results_list),eval_mol.metric_novel(results_list,genCan.observed_data)))
    fig, ax = pl.subplots(1, 1, tight_layout=True)
    eval_mol.draw_dist([res['y'] for res in res_list_filtered], ax, 'r', 'gen')
    pl.legend(loc='upper right')
    pl.xlim(xmin=-10,xmax=5)
    pl.title("max=%s,min=%s"%(max([res['y'] for res in res_list_filtered]),min([res['y'] for res in res_list_filtered])))
    pl.savefig("results/distribution_of_property_gen_%s_%s_%s_%s.png"%(model_name,acq_type,seed,count))
    pl.close()


#test
if __name__ == "__main__":
    
        input_dim_n=dataset_info(dataset_name)["num_node_type"]
        input_dim_e=dataset_info(dataset_name)["num_edge_type"]
        
        fixed_Z = torch.randn(1000,input_dim)

        #models
        encoder=Encoder(input_dim_n, input_dim_e, em_node_mlp=[57], em_edge_mlp=[57], node_mlp=[57], edge_mlp=[57], num_fine=5, encoder_out_dim=5*57, dropout=dropout, encoder_act=1, device=device)
        predictor=Predictor(encoder_out_dim=5*57, mlp_pre=[55,55,55,55,55], predictor_act=1, dropout=dropout, device=device)
        surrogate=DeepSurrogate(encoder, predictor)
        
        generator=Generator_DeConv(max_nodes=max_nodes, input_dim=82, num_node_type=dataset_info(dataset_name)["num_node_type"], num_edge_type=dataset_info(dataset_name)["num_edge_type"], channels = [64,32,32,1], kernels=[3,3,3,3], strides=[(2,1),(2,3),(1,2),(1,2)], paddings=[(1,0),(1,1),(1,1),(1,2)], output_padding=[(1,0),(1,1),(0,1),(0,0)], act=1, dropout=0.0, dataset=dataset_name, device=device)

        vaeencoder = None
        
        discriminator = []

        trainVAE = TrainVAE(vaeencoder, generator)
        trainSurr = TrainSurrogate(surrogate)
        trainGen = TrainGenerator(None, surrogate, discriminator,lam_1 = lam_1,lam_2 = lam_2,lam_3 = lam_3) #lam_1 = 0.019760427,lam_2 = 0.735347235,lam_3 = 0.122988917
        genCan = GenCandidates(dataset=dataset_name)
        chooseNext = ChooseNext(surrogate, constraint=None, generator=generator, vaeencoder=vaeencoder)
        
        eval_mol = testGenerativeQuality(filename=store_file_name,target='joint', dataset=dataset_name)
        
        
        dataset = get_data(dataset_name)
        start_idx = random.choice(range(len(dataset)-init_num))
        dataset.cut(start_idx,start_idx+init_num)
        
        
        #evaluate the init graphs
        for i in range(len(dataset)):
            results = eval_mol.test_NASBench201([[dataset[i].x,dataset[i].edge_index,dataset[i].edge_attr]],store_flag=True)
            dataset[i].y[:] = results[0]["y"]
            dataset.data_list[i].y[:] = results[0]["y"]
            #res.append(results[0]["y"])

        genCan.init_observed_data(dataset)
        
    
        generator_orig = copy.deepcopy(generator)


        #do optimize
        for iter_idx in range(max_iter):
            
            if iter_idx%retrain_step == 0:
                surrogate.reset_parameters()
                tau=1.
                lengthscale = 1e-2
                reg = lengthscale**2 * (1 - dropout) / (2. * len(dataset) * tau)
                
                trainSurr.train(dataset, batch_size=len(dataset), learning_rate=1e-3, weight_decay=reg, seed=seed, num_epochs=surr_num_epochs, convergence_err=SMALL_NUMBER)
                #trainSurr.eval_surrogate(valid=dataset_test,name="test",tau=tau)
                #trainSurr.eval_surrogate(valid=dataset,name="train",tau=tau)

                #update current generator curr_g
                if iter_idx==0:
                    num_epochs_gen_curr = gen_num_epochs0
                else:
                    num_epochs_gen_curr = gen_num_epochs1
                
                trainGen.train_GAN_and_Exp(generator, dataset, batch_size=50, lr_gen=1e-4, lr_disc=0.00005, weight_decay=1e-5, temperature=0.1, temperature_annealing_rate=1., num_epochs_gen=num_epochs_gen_curr, num_epochs_disc=5, use_random=use_random, use_regularize=False, use_exp=True, vaeencoder=vaeencoder, surrogate=surrogate, params=None)
                #if iter_idx%40==0:
                #    draw_dis(generator,fixed_Z,iter_idx)

            candidates_new = genCan.generate_candidates(generator, filter_flag=True, num_gen=1000, use_random=use_random)
            #candidates_orig = genCan.generate_candidates(generator_orig, filter_flag=True, num_gen=1000, use_random=use_random)
            candidates_orig = []
            
            candidates = candidates_new + candidates_orig
            
            print("gen=",len(candidates_new),"orig=",len(candidates_orig),"total=",len(candidates))
            chooseNext.trainGen=trainGen
            chooseNext.dataset=dataset
            chooseNext.y_observed = genCan.observed_data["y"]
            next_g_1, idx_1 = chooseNext.next(candidates,"EI")
            strategy_no = 0

            next_g = next_g_1
            evaluation_list = eval_mol.test_NASBench201(next_g)

            genCan.update_observed_data(next_g,evaluation_list)
            for i in range(len(evaluation_list)):
                if evaluation_list[i]["y"] != -10.:
                    augment_into(dataset,{'x':next_g[i][0].float(),'edge_index':next_g[i][1].long(),'edge_attr':next_g[i][2].float(),'y':torch.tensor([evaluation_list[i]["y"]]).view(-1).float()}, comments="add one")
            #print(dataset,eval_mol.denormalize([evaluation_list[i]["y"] for i in range(len(evaluation_list))]))
            #generator.save_model("models/iter_%s_%s_%s_%s"%(model_name,acq_type,seed,iter_idx+1))







