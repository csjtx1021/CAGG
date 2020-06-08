"""
  Harness for visualising a neural network.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

import functools
import graphviz as gv
import os
import networkx as nx
import numpy as np

# Parameters for plotting
_SAVE_FORMAT = 'eps'
# _SAVE_FORMAT = 'png'
_LAYER_SHAPE = 'rectangle'
_IPOP_SHAPE = 'circle'
_LAYER_FONT = 'DejaVuSans'
_IPOP_FONT = 'Helvetica'
_LAYER_FONTSIZE = '16'
_FILLCOLOR = 'transparent'
_IPOP_FONTSIZE = '12'
_IPOP_FILLCOLOR = '#ffc0cb'
_DECISION_FILLCOLOR = '#98fb98'
_GRAPH_STYLES = {
  'graph': {
    'fontsize': _LAYER_FONTSIZE,
    'rankdir': 'TB',
    'label': None,
  },
  'nodes': {
  },
  'edges': {
    'arrowhead': 'open',
    'fontsize': '12',
  }
}

GV_GRAPH = functools.partial(gv.Graph, format=_SAVE_FORMAT)
GV_DIGRAPH = functools.partial(gv.Digraph, format=_SAVE_FORMAT)

# Utilities for adding nodes, edges and styles -------------------------------------------
def add_nodes(graph, nodes):
  """ Adds nodes to the graph. """
  for n in nodes:
    if isinstance(n, tuple):
      graph.node(n[0], **n[1])
    else:
      graph.node(n)
  return graph

def add_edges(graph, edges):
  """ Adds edges to the graph. """
  # pylint: disable=star-args
  for e in edges:
    if isinstance(e[0], tuple):
      graph.edge(*e[0], **e[1])
    else:
      graph.edge(*e)
  return graph

def apply_styles(graph, styles):
  """ Applies styles to the graph. """
  graph.graph_attr.update(
      ('graph' in styles and styles['graph']) or {}
  )
  graph.node_attr.update(
      ('nodes' in styles and styles['nodes']) or {}
  )
  graph.edge_attr.update(
      ('edges' in styles and styles['edges']) or {}
  )
  return graph

# Wrappers for tedious routines ----------------------------------------------------------
def _get_ip_layer(layer_idx):
  """ Returns a tuple representing the input layer. """
  return (str(layer_idx), {'label': 'i/p', 'shape': 'circle', 'style': 'filled',
                           'fillcolor': _IPOP_FILLCOLOR, 'fontsize': _IPOP_FONTSIZE,
                           'fontname': _IPOP_FONT})

def _get_op_layer(layer_idx):
  """ Returns a tuple representing the output layer. """
  return (str(layer_idx), {'label': 'o/p', 'shape': 'circle', 'style': 'filled',
                           'fillcolor':  _IPOP_FILLCOLOR, 'fontsize': _IPOP_FONTSIZE,
                           'fontname': _IPOP_FONT})

def _get_layer(layer_idx, nn, for_pres):
  """ Returns a tuple representing the layer label. """
  if nn.layer_labels[layer_idx] in ['ip', 'op']:
    fill_colour = _IPOP_FILLCOLOR
  elif nn.layer_labels[layer_idx] in ['softmax', 'linear']:
    fill_colour = _DECISION_FILLCOLOR
  else:
    fill_colour = _FILLCOLOR
  label = nn.get_layer_descr(layer_idx, for_pres)
  return (str(layer_idx), {'label': label, 'shape': 'rectangle', 'fillcolor': fill_colour,
                           'style': 'filled', 'fontname': _LAYER_FONT}),((layer_idx), nn.layer_labels[layer_idx],(nn.num_units_in_each_layer[layer_idx]))





def _get_edge(layer_idx_start, layer_idx_end):
  """ Returns a tuple which is an edge. """
  return (str(layer_idx_start), str(layer_idx_end))

def _get_edges(conn_mat):
  """ Returns all edges. """
  starts, ends = conn_mat.nonzero()
  return [_get_edge(starts[i], ends[i]) for i in range(len(starts))]

# Main API ------------------------------------------------------------------------------
def visualise_nn(nn, save_file_prefix, fig_label=None, for_pres=True):
  """ The main API which will be used to visualise the network. """
  # First create nodes in the order
  nodes = [_get_layer(i, nn, for_pres)[0] for i in range(nn.num_layers)]
  nodes_my = [_get_layer(i, nn, for_pres)[1] for i in range(nn.num_layers)]
  #print("nodes_my=",nodes_my)
  edges = _get_edges(nn.conn_mat)
  edges_my = [(int(s),int(t)) for s,t in edges]
  #print("edges_my=",edges_my)
  nn_graph = GV_DIGRAPH()
  add_nodes(nn_graph, nodes)
  add_edges(nn_graph, edges)
  graph_styles = _GRAPH_STYLES
  graph_styles['graph']['label'] = fig_label
  apply_styles(nn_graph, graph_styles)
  nn_graph.render(save_file_prefix)
  
  if os.path.exists(save_file_prefix):
    # graphviz also creates another file in the name of the prefix. delete it.
    os.remove(save_file_prefix)

  return tonxgraph(nodes_my,edges_my)

NODE_TYPES = ['ip', 'op', 'linear']
hidden_list = [8,16,32,64,128,256,512,1024]
for i in hidden_list:
    NODE_TYPES.append("relu-%s"%i)
    NODE_TYPES.append("crelu-%s"%i)
    NODE_TYPES.append("leaky-relu-%s"%i)
    NODE_TYPES.append("softplus-%s"%i)
    NODE_TYPES.append("elu-%s"%i)
    NODE_TYPES.append("logistic-%s"%i)
    NODE_TYPES.append("tanh-%s"%i)


def tonxgraph(nodes_my,edges_my):
    g = {"x":[],"edge_index":[],"edge_attr":[]}
    
    for n_idx, type, num_hidden in nodes_my:
        n_idx = int(n_idx)
        if type=='ip' or type=='op' or type=='linear':
            g["x"].append(np.eye(len(NODE_TYPES))[NODE_TYPES.index(type)])
        else:
            num_hidden = np.random.choice(hidden_list)
            g["x"].append(np.eye(len(NODE_TYPES))[NODE_TYPES.index("%s-%s"%(type,num_hidden))])
    row = []
    col = []
    for s, t in edges_my:
        row.append(s)
        col.append(t)
        g["edge_attr"].append(np.ones(1))
    g["edge_index"].append(row)
    g["edge_index"].append(col)

    g["x"]=np.array(g["x"])
    g["edge_attr"]=np.array(g["edge_attr"])

    print("+",g["x"].shape)
    assert g["x"].shape[0] <= 20
    
    return g




    #g_nx = nx.nx_agraph.from_agraph(nn_graph)
    #A = nx.nx_agraph.to_agraph(g_nx)        # convert to a graphviz graph
    #A.layout()            # neato layout
    #A.draw("a.ps")

def visualise_list_of_nns(list_of_nns, save_dir, fig_labels=None, fig_file_names=None,
                          for_pres=False):
  """ Visualises a list of neural networks. """
  g_list = []
  if fig_labels is None:
    fig_labels = [None] * len(list_of_nns)
  if fig_file_names is None:
    fig_file_names = [str(idx) for idx in range(len(list_of_nns))]
  for idx, nn in enumerate(list_of_nns):
    save_file_prefix = os.path.join(save_dir, fig_file_names[idx])
    g = visualise_nn(nn, save_file_prefix, fig_labels[idx], for_pres)
    g_list.append(g)
  return g_list

