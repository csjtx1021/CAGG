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
from data_utils import get_data, to_mol, dataset_info
import csv
import seaborn as sns
import tools



#------------------------------------------------------------
#to test the Generative Quality
#------------------------------------------------------------
class testGenerativeQuality():
    def __init__(self, filename="results/observations_mcdropout.csv", target='joint', dataset='qm9'):
        self.target = target
        self.dataset = dataset
        self.filename = filename
        self.count = 0

    #recall the real evaluation function to evaluate
    def evaluate_point(self, input):
        if input["smiles"]=="":
            return -10.0
        else:
            y=objective_func.evaluate_point(input,target=self.target)
            return y

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
    
    def test_dataset(self,dataset):
        res = []
        for i in range(len(dataset)):
            res.append(dataset[i].y[:,TASK_NO].item())
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
            if re["smiles"] not in observed_data:
                count += 1.
        return float(count)/len(res_list_filtered)


#test
if __name__ == "__main__":
    
    dataset_name="qm9"
    
    property_name="logppunishedbysa" # "qed" "logp" "logppunishedbysa"
    
    store_file_name = "results/observations_%s_%s.csv"%(dataset_name,property_name)

    #make data
    dataset = get_data(dataset_name)
    #dataset.cut(0,init_num)
    
    eval_mol = testGenerativeQuality(filename=store_file_name, target=property_name, dataset=dataset_name)
    
    #evaluate the init graphs
    for i in range(len(dataset)):
        results = eval_mol.test([[dataset[i].x,dataset[i].edge_index,dataset[i].edge_attr]],store_flag=True)
        #dataset[i].y[:,TASK_NO] = results[0]["y"]
        #dataset.data_list[i].y[:,TASK_NO] = results[0]["y"]
        if i%500==0:
            print(i+1,"/",len(dataset),end="\r")




