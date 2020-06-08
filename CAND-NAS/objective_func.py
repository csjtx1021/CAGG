#!/usr/bin/env/python
from __future__ import print_function, division

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Tanh, LeakyReLU, BatchNorm1d as BN1
from torch.utils.data import Dataset,DataLoader
from torch.nn import functional as F, MSELoss, BCELoss
import data_utils
import pickle as cPickle
import time
import numpy as np
import os
import sys

mlp_dataset_name="data/IndoorLoc.p" #input_dim=521
#mlp_dataset_name="data/SliceLocalization.p" #input_dim=385

#mlp_all_data = cPickle.load(open(mlp_dataset_name, 'rb'),encoding='latin1')
#print(mlp_all_data["train"]["x"].shape,len(mlp_all_data["vali"]["x"]),len(mlp_all_data["test"]["x"]))

#exit(1)
def split_dataset(mlp_dataset_name):
    if "IndoorLoc" in mlp_dataset_name:
        mlp_all_data = cPickle.load(open(mlp_dataset_name, 'rb'),encoding='latin1')
        
        all_data_x = np.concatenate([mlp_all_data["train"]["x"],mlp_all_data["vali"]["x"]],0)
        all_data_y = np.concatenate([mlp_all_data["train"]["y"],mlp_all_data["vali"]["y"]],0)

        train_data_x = all_data_x[0:int(len(all_data_y)*0.6)]
        train_data_y = all_data_y[0:int(len(all_data_y)*0.6)]

        vali_data_x = all_data_x[int(len(all_data_y)*0.6):int(len(all_data_y)*0.8)]
        vali_data_y = all_data_y[int(len(all_data_y)*0.6):int(len(all_data_y)*0.8)]

        test_data_x = all_data_x[int(len(all_data_y)*0.8):]
        test_data_y = all_data_y[int(len(all_data_y)*0.8):]

        input_dim = train_data_x.shape[1]

    elif "SliceLocalization" in mlp_dataset_name:
        mlp_all_data = cPickle.load(open(mlp_dataset_name, 'rb'),encoding='latin1')

        train_data_x = mlp_all_data["train"]["x"]
        train_data_y = mlp_all_data["train"]["y"]

        vali_data_x = mlp_all_data["vali"]["x"]
        vali_data_y = mlp_all_data["vali"]["y"]

        test_data_x = mlp_all_data["test"]["x"]
        test_data_y = mlp_all_data["test"]["y"]

        input_dim = train_data_x.shape[1]
    
    else:
        print("wrong name :%s"%mlp_dataset_name)
        exit(1)

    return input_dim,train_data_x,train_data_y,vali_data_x,vali_data_y,test_data_x,test_data_y

input_dim,train_data_x,train_data_y,vali_data_x,vali_data_y,test_data_x,test_data_y = split_dataset(mlp_dataset_name)

class MyDataset(Dataset):
    def __init__(self, mlp_dataset_name, train_vali):
        if train_vali == "train":
            self.mlp_all_data = {"x":train_data_x,"y":train_data_y}
        elif train_vali == "vali":
            self.mlp_all_data = {"x":vali_data_x,"y":vali_data_y}
        elif train_vali == "test":
            self.mlp_all_data = {"x":test_data_x,"y":test_data_y}
        else:
            print("wrong %s"%train_vali)
            exit(1)

    def __getitem__(self, index):
        return self.mlp_all_data['x'][index], self.mlp_all_data['y'][index]

    def __len__(self):
        return len(self.mlp_all_data['x'])


def get_default_mlp_tf_params():
  """ Default MLP training parameters for tensorflow. """
  return {
    'trainBatchSize':256,
    'valiBatchSize':1000,
    'trainNumStepsPerLoop':100,
    'valiNumStepsPerLoop':5,
    'numLoops':200,
    'learningRate':0.00001,
    }

#real evaluation function
def evaluate_point(input_g):

    train_data = MyDataset(mlp_dataset_name,'train')
    print(len(train_data))
    vali_data = MyDataset(mlp_dataset_name,'vali') 
    print(len(vali_data))
    test_data = MyDataset(mlp_dataset_name,'test') 
    print(len(test_data))
    #exit(1)
    train_params = get_default_mlp_tf_params()
    model = Graph2TorchNN(input_dim, input_g)
    optimizer = torch.optim.SGD(model.parameters(), lr=train_params["learningRate"])
    
    loss_fn = MSELoss()
    train_loss_store = []
    vali_loss_store = []
    train_time=time.time()
    for epoch in range(0, train_params["numLoops"]):
        optimizer.zero_grad()
        data_loader = DataLoader(train_data, batch_size=train_params["trainBatchSize"], shuffle=True)
        #dl_data = iter(data_loader)

        train_num = 0
        train_loss_value = 0
        #for step in range(0, train_params["trainNumStepsPerLoop"]):
        for batch_data in data_loader:
            #batch_data = next(dl_data)
            train_num += batch_data[0].size(0)

            y_real = batch_data[1]
            optimizer.zero_grad()
            y_pre = model(batch_data[0]).view(-1)

            train_loss = loss_fn(y_real,y_pre)
            
            train_loss.backward()
            optimizer.step()

            train_loss_value += train_loss.item()*batch_data[0].size(0)
        print("training epoch %s: loss = %s, time cost=%s"%(epoch,train_loss_value/float(train_num),(time.time()-train_time)),end="\r")
        train_loss_store.append(train_loss_value/float(train_num))

        vali_data_loader = DataLoader(vali_data, batch_size=train_params["valiBatchSize"], shuffle=False)
        #vali_dl_data = iter(vali_data_loader)
        vali_num = 0
        vali_loss_value = 0
        #for step in range(0, train_params["valiNumStepsPerLoop"]): 
        for vali_batch_data in vali_data_loader: 
            #vali_batch_data = next(vali_dl_data)
            vali_num += vali_batch_data[0].size(0)
            vali_y_real = vali_batch_data[1]
            vali_y_pre = model(vali_batch_data[0]).view(-1)

            vali_loss = loss_fn(vali_y_real,vali_y_pre)

            vali_loss_value += vali_loss.item()*vali_batch_data[0].size(0)
        vali_loss_store.append(vali_loss_value/float(vali_num))
        print("vali epoch %s: loss = %s, time cost=%s"%(epoch,vali_loss_value/float(vali_num),(time.time()-train_time)),end="\r")
    res = -1.0*min(vali_loss_store)
    if np.isnan(res):
        res = -10.
    return res


def act(act_name):
    if act_name=="linear":
        return None
    elif act_name=="relu":
        return torch.nn.ReLU()
    elif act_name=="crelu":
        return CRELU()
    elif act_name=="leaky-relu":
        return torch.nn.LeakyReLU()
    elif act_name=="softplus":
        return torch.nn.Softplus()
    elif act_name=="elu":
        return torch.nn.ELU()
    elif act_name=="logistic":
        return torch.nn.Sigmoid()
    elif act_name=="tanh":
        return torch.nn.Tanh()
    else:
        print("error act_name %s"%act_name)
        exit(1)

class CRELU(torch.nn.Module):
    def __init__(self, ):
        super(CRELU, self).__init__()
    def forward(self,in_feat):
        m = torch.nn.ReLU()
        return torch.cat([m(in_feat),m(-in_feat)],dim=-1)

class Graph2TorchNN(torch.nn.Module):
    def __init__(self, input_dim, g):
        super(Graph2TorchNN, self).__init__()
        self.x=g[0]
        self.edge_index=g[1]
        self.edge_attr=g[2]

        #nodes index
        self.nodes = [int(idx) for idx in torch.sum(self.x,-1).nonzero().view(-1).numpy()]
        #find father and child
        self.father = {}
        self.child = {}
        row, col = self.edge_index
        for i in self.nodes:
            self.father[i]=[int(idx) for idx in row[col==i].numpy()]
            self.child[i]=[int(idx) for idx in col[row==i].numpy()]
        #make nodes(layers)
        self.layers = {}
        self.layers_do = torch.nn.ModuleDict()
        
        for idx in self.nodes:
            nodetype = data_utils.NODE_TYPES[self.x[idx].nonzero().view(-1).item()]
            if nodetype["type"] == 'ip':
                self.layers[0]={"idx":0,"in_dim":input_dim,"out_dim":input_dim,"in_feat":[],"out_feat":None}
            else:
                in_dim = sum([self.layers[fa_idx]["out_dim"] for fa_idx in self.father[idx]])
                self.layers_do[str(idx)] = torch.nn.Sequential()
                if nodetype["type"] == "linear" or nodetype["type"] == "op":
                    self.layers_do[str(idx)].add_module("layer_%s_linear"%idx,Lin(in_dim, nodetype["hidden_num"]))
                else:
                    self.layers_do[str(idx)].add_module("layer_%s_linear"%idx,Lin(in_dim, nodetype["hidden_num"]))
                    self.layers_do[str(idx)].add_module("layer_%s_act"%idx,act(nodetype["type"]))
                if nodetype["type"] == "crelu":
                    out_dim = int(nodetype["hidden_num"]*2)
                else:
                    out_dim = nodetype["hidden_num"]
                self.layers[idx]={"idx":idx,"in_dim":in_dim,"out_dim":out_dim,"in_feat":[],"out_feat":None}

    def clear_in_feat(self,):
        for idx in self.nodes:
            self.layers[idx]["in_feat"]=[]

    def forward(self, input_feat):
        self.clear_in_feat()
        #connect
        for idx in self.nodes:
            #do operation, get output feature
            if idx==0:
                self.layers[idx]["out_feat"] = input_feat.float()
            else:
                self.layers[idx]["out_feat"] = self.layers_do[str(idx)](torch.cat(self.layers[idx]["in_feat"],dim=-1))
            #send output feature to children's input
            children = self.child[idx]
            for chi in children:
                self.layers[chi]["in_feat"].append(self.layers[idx]["out_feat"])
        return self.layers[self.nodes[-1]]["out_feat"] #return op
    

#test a optimal network
def test_a_net(filename):
    pkl_file = open(filename, 'rb')
    input_g = cPickle.load(pkl_file,encoding='latin1')
    
    print(input_g)

    train_params = get_default_mlp_tf_params()

    train_data = MyDataset(mlp_dataset_name,'train')
    print(len(train_data))
    vali_data = MyDataset(mlp_dataset_name,'vali') 
    print(len(vali_data))
    test_data = MyDataset(mlp_dataset_name,'test') 
    print(len(test_data))

    model = Graph2TorchNN(input_dim, input_g)
    optimizer = torch.optim.SGD(model.parameters(), lr=train_params["learningRate"])
    
    loss_fn = MSELoss()
    train_loss_store = []
    vali_loss_store = []
    test_loss_store = []
    train_time=time.time()
    for epoch in range(0, train_params["numLoops"]):
        optimizer.zero_grad()
        data_loader = DataLoader(train_data, batch_size=train_params["trainBatchSize"], shuffle=True)
        #dl_data = iter(data_loader)

        train_num = 0
        train_loss_value = 0
        #for step in range(0, train_params["trainNumStepsPerLoop"]):
        for batch_data in data_loader:
            #batch_data = next(dl_data)
            train_num += batch_data[0].size(0)

            y_real = batch_data[1]
            optimizer.zero_grad()
            y_pre = model(batch_data[0]).view(-1)

            train_loss = loss_fn(y_real,y_pre)
            
            train_loss.backward()
            optimizer.step()

            train_loss_value += train_loss.item()*batch_data[0].size(0)
        print("training epoch %s: loss = %s, time cost=%s"%(epoch,train_loss_value/float(train_num),(time.time()-train_time)),end="\n")
        train_loss_store.append(train_loss_value/float(train_num))

        vali_data_loader = DataLoader(vali_data, batch_size=train_params["valiBatchSize"], shuffle=False)
        #vali_dl_data = iter(vali_data_loader)
        vali_num = 0
        vali_loss_value = 0
        #for step in range(0, train_params["valiNumStepsPerLoop"]): 
        for vali_batch_data in vali_data_loader: 
            #vali_batch_data = next(vali_dl_data)
            vali_num += vali_batch_data[0].size(0)
            vali_y_real = vali_batch_data[1]
            vali_y_pre = model(vali_batch_data[0]).view(-1)

            vali_loss = loss_fn(vali_y_real,vali_y_pre)

            vali_loss_value += vali_loss.item()*vali_batch_data[0].size(0)
        vali_loss_store.append(vali_loss_value/float(vali_num))
        print("vali epoch %s: loss = %s, time cost=%s"%(epoch,vali_loss_value/float(vali_num),(time.time()-train_time)),end="\n")

        test_data_loader = DataLoader(test_data, batch_size=train_params["valiBatchSize"], shuffle=False)
        #vali_dl_data = iter(vali_data_loader)
        test_num = 0
        test_loss_value = 0
        #for step in range(0, train_params["valiNumStepsPerLoop"]): 
        for test_batch_data in test_data_loader: 
            #vali_batch_data = next(vali_dl_data)
            test_num += test_batch_data[0].size(0)
            test_y_real = test_batch_data[1]
            test_y_pre = model(test_batch_data[0]).view(-1)

            test_loss = loss_fn(test_y_real,test_y_pre)

            test_loss_value += test_loss.item()*test_batch_data[0].size(0)
        test_loss_store.append(test_loss_value/float(test_num))
        print("test epoch %s: loss = %s, time cost=%s"%(epoch,test_loss_value/float(test_num),(time.time()-train_time)),end="\n")

    return test_loss_store[np.argmin(vali_loss_store)]
    

#test

if __name__ == "__main__":
    #dataset = data_utils.get_data("nn")
    #input_g = [dataset[0].x,dataset[0].edge_index,dataset[0].edge_attr]
    #res = evaluate_point(input_g)

    #filename="results/architecture_28_1.pkl"
    #filename="results/architecture_ours-soft-decode6_slice_31_1.pkl"
    #filename="results/architecture_ours-soft-decode6_slice_13_1.pkl"
    filename = sys.argv[1]
    res = test_a_net(filename)
    print("error:",res)

