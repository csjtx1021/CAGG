"""
  Unit tests for nn_examples.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name

import os
import numpy as np
import pickle


# Local imports
from utils.base_test_class import BaseTestClass, execute_tests
from nn import nn_examples
from nn.nn_visualise import visualise_list_of_nns

# Test cases for nn_examples.py ----------------------------------------------------------
class NNExamplesTestCase(BaseTestClass):
  """ Unit test for some neural network examples. We are just testing for generation. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNExamplesTestCase, self).__init__(*args, **kwargs)
    self.save_dir = '../scratch/unittest_examples'

  def test_vgg(self):
    """ Unit test for the VGG_net."""
    self.report('Testing the VGG net. ')
    nn_examples.get_vgg_net(2)

  def test_blocked_cnn(self):
    """ Unit test for a blocked CNN. """
    self.report('Testing a blocked CNN.')
    nn_examples.get_blocked_cnn(4, 4, 1)

  def test_generate_many_nns(self):
    """ Testing generation of many neural networks. """
    self.report('Testing generation of many NNs.')
    num_nns = 1000
    #cnns = nn_examples.generate_many_neural_networks('cnn', num_nns)
    #visualise_list_of_nns(cnns, os.path.join(self.save_dir, 'cnn'))
    reg_mlps = nn_examples.generate_many_neural_networks('mlp-reg', num_nns)
    g_list = visualise_list_of_nns(reg_mlps, os.path.join(self.save_dir, 'reg_mlps'))

    #print(g_list)
    with open('randNN1000.pkl', 'w') as f:
        pickle.dump(g_list, f)
    
    pkl_file = open('randNN1000.pkl', 'r')
    data1 = pickle.load(pkl_file)

    print("Generated %s neural networks"%len(data1))
    #class_mlps = nn_examples.generate_many_neural_networks('mlp-class', num_nns)
    #visualise_list_of_nns(class_mlps, os.path.join(self.save_dir, 'class_mlps'))


if __name__ == '__main__':
  execute_tests()

