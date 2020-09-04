from __future__ import print_function, division

import random

from nas_201_api import NASBench201API as API

#get_cell_based_tiny_net
import sys
sys.path.insert(0, '/Users/cui-pro/Desktop/2019/GNN/A2G2-ActiveAttributedGraphGeneration/ActiveGG-master-Solution4-GraphDesign-NAS/AutoDL-Projects-master/lib')

from models import get_cell_based_tiny_net # this module is in AutoDL-Projects/lib/models


from optparse import OptionParser
parser = OptionParser()
parser.add_option("-s", "--str", dest="arch_str", default=None)
parser.add_option("-f", "--file", dest="file_name", default=None)

opts,args = parser.parse_args()

arch_str = opts.arch_str
file_name = opts.file_name

def query(arch_str):

    api = API('NAS-Bench-201-v1_0-e61699.pth')
    #api = API('NAS-Bench-201-v1_1-096897.pth')

    #num = len(api)
    #for i, arch_str in enumerate(api):
    #  print ('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))

    #config = api.get_net_config(123, 'cifar10') # obtain the network configuration for the 123-th architecture on the CIFAR-10 dataset
    #network = get_cell_based_tiny_net(config) # create the network from configurration
    #print(network) # show the structure of this architecture

    #arch_str = '|nor_conv_3x3~0|+|nor_conv_1x1~0|avg_pool_3x3~1|+|skip_connect~0|avg_pool_1x1~1|nor_conv_3x3~2|'
    #arch_str = '|nor_conv_3x3~0|+|none~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_1x1~1|skip_connect~2|'
    #arch_str = '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|'
    
    index = api.query_index_by_arch(arch_str)
    print("index = ",index)
    # get the detailed information
    results = api.query_by_index(index, 'cifar100') # a dict of all trials for 1st net on cifar100, where the key is the seed
    print ('There are {:} trials for this architecture [{:}] on cifar100'.format(len(results), api[index]))
    trial_seed_1 = list(results.keys())[0]
    print ('Latency : {:}'.format(results[trial_seed_1].get_latency()))
    print ('Train Info : {:}'.format(results[trial_seed_1].get_train()))
    print ('Valid Info : {:}'.format(results[trial_seed_1].get_eval('x-valid')))
    print ('Test  Info : {:}'.format(results[trial_seed_1].get_eval('x-test')))
    # for the metric after a specific epoch
    print ('Train Info [10-th epoch] : {:}'.format(results[trial_seed_1].get_train(10)))




OPS    = ['zeroize', 'skip-connect', 'conv-1x1', 'conv-3x3', 'pool-3x3']
#COLORS = ['gray', 'chartreuse'  , 'cyan'    , 'navyblue', 'chocolate1']
COLORS = ['gray25', 'mediumseagreen', 'deepskyblue', 'purple3', 'darkorange2']

def get_ops_idx(op_str):
    if op_str == "none":
        return 0
    elif op_str == "skip_connect":
        return 1
    elif op_str == "nor_conv_1x1":
        return 2
    elif op_str == "nor_conv_3x3":
        return 3
    elif op_str == "avg_pool_3x3":
        return 4
    else:
        print("error op_str [%s]"%op_str)
        exit(1)

def plot(filename,arch_str):
  from graphviz import Digraph
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  steps = 4
  for i in range(0, steps):
    if i == 0:
      g.node(str(i), fillcolor='darkseagreen2')
    elif i+1 == steps:
      g.node(str(i), fillcolor='palegoldenrod')
    else: g.node(str(i), fillcolor='lightblue')

  for i in range(1, steps):
    xin_str = arch_str.split('+')[i-1].split('|')[1:-1]
    for xin in range(i):
      if int(xin_str[xin][-1]) == xin:
          op_i = get_ops_idx(xin_str[xin][:-2])
      else:
          print("error arch_str [%s] != [%s]"%(int(xin_str[xin][-1]),xin))
          exit(1)
      g.edge(str(xin), str(i), label=OPS[op_i], color=COLORS[op_i], fillcolor=COLORS[op_i], fontcolor=COLORS[op_i])
      #import pdb; pdb.set_trace()
  g.render(filename, cleanup=True, view=False)


plot(file_name,arch_str)
#query(arch_str)


