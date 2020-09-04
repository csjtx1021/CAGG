import torch
from torch_scatter import scatter_add


def to_dense_batch(x, max_num_nodes=None, batch=None, fill_value=0):
   
    if batch is None:
        mask = torch.ones(1, x.size(0), device=x.device)
        return x.unsqueeze(0), mask
    
    batch_size = batch[-1].item() + 1
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0,dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    b=batch_size
    n=max_num_nodes

    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
                            
    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])
                            
    mask = torch.ones(batch_size * max_num_nodes,device=x.device)
    mask[idx] = 0
    mask = mask.view(batch_size, max_num_nodes, 1)
                            
    return torch.cat([mask, out],dim=-1).view(b*n,-1) #[b*n,1+d]


def to_dense_adj(edge_index, max_num_nodes=None, batch=None, edge_attr=None):

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)
    batch_size = batch[-1].item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_add(one, batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()
    
    size = [batch_size, max_num_nodes, max_num_nodes]
    size = size if edge_attr is None else size + list(edge_attr.size())[1:]
    size[-1]=size[-1]+1
    dtype = torch.float if edge_attr is None else edge_attr.dtype
    adj = torch.zeros(size, dtype=dtype, device=edge_index.device)
    
    edge_index_0 = batch[edge_index[0]].view(1, -1)
    edge_index_1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    edge_index_2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if edge_attr is None:
        adj[edge_index_0, edge_index_1, edge_index_2] = 1
    else:
        init_attr = torch.zeros(1+edge_attr.size(-1),dtype=dtype, device=edge_index.device) #init
        init_attr[0] = 1.
        adj[:, :, :] = init_attr
        adj[edge_index_0, edge_index_1, edge_index_2] = torch.cat([torch.zeros(edge_attr.size(0),1,dtype=dtype, device=edge_index.device),edge_attr],dim=-1)
    adj = adj.view(-1,max_num_nodes,1+edge_attr.size(-1))

    b=batch_size
    n=max_num_nodes
    m=n*(n-1)
    
    all_edges = torch.tensor([[i,j] for i in range(n) for j in range(i+1,n)]).long().to(edge_index.device)
    edge_index_one = all_edges.t()
    edge_index_one = edge_index_one[[0,1,1,0]].view(2,-1)
    edge_index = []
    for i in range(b):
        edge_index.append(edge_index_one+i*n)
    edge_index = torch.cat(edge_index,dim=-1).long()
    col = []
    for i in range(b):
        col.append(edge_index[1][0:m])
    col = torch.cat(col)
    attr = adj[edge_index[0],col,:]

    return attr #[b*n*(n-1),1+d]


#test
if __name__ == "__main__":
    edge_index=torch.tensor([[0,1,2,2,3,4,5,6],[1,0,3,4,2,2,6,5]]).long()
    edge_attr=torch.randn(8,3)
    batch=torch.tensor([0,0,1,1,1,2,2])
    x = to_dense_adj(edge_index, max_num_nodes=5, batch=batch, edge_attr=edge_attr)
    print(edge_attr,x,x.size())

    


