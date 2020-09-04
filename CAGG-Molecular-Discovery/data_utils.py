import os

import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar, Data)
from rdkit import Chem
import numpy as np
import random

class QM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
        Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
        about 130,000 molecules with 13 regression targets.
        Each molecule includes complete spatial information for the single low
        energy conformation of the atoms in the molecule.
        In addition, we provide the atom features from the `"Neural Message
        Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.
        
        Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
        """
    
    url = 'http://www.roemisch-drei.de/qm9.tar.gz'
    
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(QM9, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.data_list = torch.load(self.processed_paths[0])
    

    
    @property
    def raw_file_names(self):
        return 'qm9.pt'
    
    @property
    def processed_file_names(self):
        return 'data_filtered.pt'
    
    def cut(self, from_idx=0, to_idx=0):
        assert from_idx < to_idx
        self.data_list = self.data_list[from_idx:to_idx]
        self.data, self.slices = self.collate(self.data_list)
    
    def cutfromlist(self, index_list):
        assert len(index_list) > 0
        data_list_temp = []
        for idx in index_list:
            data_list_temp.append(self.data_list[idx])
        self.data_list = data_list_temp
        self.data, self.slices = self.collate(self.data_list)
    
    def shuffle(self):
        random.shuffle(self.data_list)
        self.data, self.slices = self.collate(self.data_list)
    
    def add(self,d):
        #print([i.y[0,0] for i in self.data_list])
        self.data_list.append(d)
        #print([i.y[0,0] for i in self.data_list])
        self.data, self.slices = self.collate(self.data_list)
    
    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        extract_tar(file_path, self.raw_dir, mode='r')
        os.unlink(file_path)
    
    def process(self):
        raw_data_list = torch.load(self.raw_paths[0])
        #print(raw_data_list[0])
       
        #Remove redundant dimensions and hydrogen atoms, and only retain node types, edge types and heavy atoms (up to 9)
        #print(raw_data_list[0]['x'],raw_data_list[0]['edge_index'])
        data_list = []
        count = 0
        for d in raw_data_list:
            #print("process:",{"x":d['x'][:,0:5],"edge_index":d['edge_index'],"edge_attr":d['edge_attr'][:,0:3]})
            mol = to_mol({"x":d['x'][:,0:5],"edge_index":d['edge_index'],"edge_attr":d['edge_attr'][:,0:3]}, dataset='qm9',orig_flag=True)
            if mol is None:
                continue
            
            g = to_graph(Chem.MolToSmiles(mol), 'qm9')
            #print("g=",g)
            #exit(1)
            data_list.append(Data(x=g['x'],
                                  edge_index=g['edge_index'],
                                  edge_attr=g['edge_attr'],
                                  y=d['y']))
            count+=1
            assert Chem.MolToSmiles(mol)==Chem.MolToSmiles(to_mol(g))
            print(" # processed =",count,end="\r")
                

        #print(data_list[0].x,data_list[0].edge_index,)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data,'qm9')]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        data, slices = self.collate(data_list)
        torch.save((data, slices, data_list), self.processed_paths[0])

def pre_filter(data, dataset='qm9'):
    if to_mol(data, dataset) is None:
        return False
    return True

def to_mol(graph, dataset='qm9', orig_flag=False):
    
    x = graph['x']
    edge_index = graph['edge_index']
    edge_attr = graph['edge_attr']
    
    atom_idx, atom_type = torch.where(x==1)
    
    atom_type = list(atom_type.numpy())
    
    # New molecule
    new_mol = Chem.MolFromSmiles('')
    new_mol = Chem.rdchem.RWMol(new_mol)
    # Add atoms
    add_atoms(new_mol, atom_type, dataset, orig_flag)
    
    #add the bond
    exist_edges=[]
    if len(edge_index) > 0:
        from_nodes, to_nodes = edge_index
    #print("from_nodes.size(0)=",from_nodes.size(0))
    for i in range(edge_attr.size(0)):
        from_node = int(from_nodes[i])
        to_node = int(to_nodes[i])
        edge_type = np.argmax(edge_attr[i].numpy())
    
        if np.sum(edge_attr[i].numpy())!=1 or from_node == to_node:
            return None
        
        #print("(%s) %s->%s"%(i+1,from_node,to_node))
        if (from_node,to_node) not in exist_edges:
            #print("add",from_node,to_node,edge_type)
            
            new_mol.AddBond(int(from_node), int(to_node), number_to_bond[edge_type])
            exist_edges.append((from_node,to_node))
            exist_edges.append((to_node,from_node))
        
    # Remove unconnected node
    remove_extra_nodes(new_mol)
    try:
        smi = Chem.MolToSmiles(new_mol)
    except:
        return None

    new_mol=Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

    return new_mol


def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z

def need_kekulize(mol):
    for bond in mol.GetBonds():
        if bond_dict[str(bond.GetBondType())] >= 3:
            return True
    return False

def to_graph(smiles, dataset):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Kekulize it
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return None
    # remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    x = []
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        #print("bond.GetBondType():", bond.GetBondType(), bond.GetBeginAtomIdx(), "->", bond.GetEndAtomIdx(), mol.GetAtoms())
        edge_index.append(torch.tensor([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]).view(1,-1))
        edge_index.append(torch.tensor([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]).view(1,-1))
        edge_attr.append(torch.tensor(onehot(bond_dict[str(bond.GetBondType())], dataset_info(dataset)['num_edge_type'])).view(1,-1))
        edge_attr.append(torch.tensor(onehot(bond_dict[str(bond.GetBondType())], dataset_info(dataset)['num_edge_type'])).view(1,-1))
        assert bond_dict[str(bond.GetBondType())] != 3
    for atom in mol.GetAtoms():
        #print("atom.GetSymbol()",atom.GetSymbol())
        if dataset=='qm9':
            x.append(torch.tensor(onehot(dataset_info(dataset)['atom_types'].index(atom.GetSymbol()), len(dataset_info(dataset)['atom_types']))).view(1,-1))
    if len(edge_index)==0:
        edge_index = torch.tensor([]).long().view(2,-1)
        edge_attr = torch.tensor([]).float().view(-1,dataset_info(dataset)['num_edge_type'])
    else:
        edge_index = torch.cat(edge_index,dim=0).t().long().view(2,-1)
        edge_attr = torch.cat(edge_attr,dim=0).float().view(-1,dataset_info(dataset)['num_edge_type'])
    return {"x":torch.cat(x,dim=0).float().view(-1,dataset_info(dataset)['num_node_type']),"edge_index":edge_index,"edge_attr":edge_attr}



def get_data(data="qm9"):
    if data=="qm9":
        dataset = QM9(root='./data/QM9',pre_filter=pre_filter)
    return dataset


# bond mapping
bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}
number_to_bond= {0: Chem.rdchem.BondType.SINGLE, 1:Chem.rdchem.BondType.DOUBLE,
    2: Chem.rdchem.BondType.TRIPLE, 3:Chem.rdchem.BondType.AROMATIC}

def dataset_info(dataset):
    if dataset=='qm9':
        return { 'atom_types': ["C", "N", "O", "F"],
                'maximum_valence': {0: 4, 1: 3, 2: 2, 3: 1},
                'bond': {0: 1, 1: 2, 2: 3},
                'number_to_atom': {0: "C", 1: "N", 2: "O", 3: "F"},
                'num_node_type': 4,
                'num_edge_type': 3
                }
    else:
        print("Unknown dataset %s. Please choose one from ['qm9',]."%dataset)
        exit(1)

def orig_dataset_info(dataset):
    if dataset=='qm9':
        return { 'atom_types': ["H","C", "N", "O", "F"],
                'maximum_valence': {0:1, 1: 4, 2: 3, 3: 2, 4: 1},
                'bond': {0: 1, 1: 2, 2: 3},
                'number_to_atom': {0:"H", 1: "C", 2: "N", 3: "O", 4: "F"},
                'num_node_type': 5,
                'num_edge_type': 3
            }
    else:
        print("Unknown dataset %s. Please choose one from ['qm9',]."%dataset)
        exit(1)


def add_atoms(new_mol, node_symbol, dataset, orig_flag=False):
    for number in node_symbol:
        if dataset=='qm9' or dataset=='cep':
            if orig_flag:
                idx=new_mol.AddAtom(Chem.Atom(orig_dataset_info(dataset)['number_to_atom'][number]))
            else:
                idx=new_mol.AddAtom(Chem.Atom(dataset_info(dataset)['number_to_atom'][number]))
        elif dataset=='zinc':
            if orig_flag:
                new_atom = Chem.Atom(orig_dataset_info(dataset)['number_to_atom'][number])
                charge_num=int(orig_dataset_info(dataset)['atom_types'][number].split('(')[1].strip(')'))
                new_atom.SetFormalCharge(charge_num)
                new_mol.AddAtom(new_atom)
            else:
                new_atom = Chem.Atom(dataset_info(dataset)['number_to_atom'][number])
                charge_num=int(dataset_info(dataset)['atom_types'][number].split('(')[1].strip(')'))
                new_atom.SetFormalCharge(charge_num)
                new_mol.AddAtom(new_atom)

def get_idx_of_largest_frag(frags):
    return np.argmax([len(frag) for frag in frags])

def remove_extra_nodes(new_mol):
    frags=Chem.rdmolops.GetMolFrags(new_mol)
    while len(frags) > 1:
        # Get the idx of the frag with largest length
        largest_idx = get_idx_of_largest_frag(frags)
        for idx in range(len(frags)):
            if idx != largest_idx:
                # Remove one atom that is not in the largest frag
                new_mol.RemoveAtom(frags[idx][0])
                break
        frags=Chem.rdmolops.GetMolFrags(new_mol)
