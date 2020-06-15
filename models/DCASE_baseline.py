#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy

class AutoPool(nn.Module):
   """ Adaptive pooling operators for Multiple Instance Learning
   Adapted original code.
   This layer automatically adapts the pooling behavior to interpolate
   between min, mean and max-pooling for each class.
   Link : https://github.com/marl/autopool
   Args:
      input_size (int): Lenght of input_vector
      time_axis (int): Axis along which to perform the pooling. 
         By default 1 (should be time) ie. (batch_size, time_sample_size, input_size)
   """
   
   def __init__(self, input_size, time_axis=1):
      super(AutoPool, self).__init__()
      self.time_axis = time_axis
      self.alpha = nn.Parameter(torch.zeros(1, input_size))

   def forward(self, x):
      scaled = self.alpha*x
      weights = F.softmax(scaled, dim=self.time_axis)
      return (x * weights).sum(dim=self.time_axis)

class TimeDistributed(nn.Module):
   """ Takes an operation and applies it for each time sample
   Ref: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
   Args:
      module (nn.Module): The operation
      batch_first (bool): If true, x is (samples, timesteps, output_size)
         Else, x is (timesteps, samples, output_size)
   """

   def __init__(self, module, batch_first=True):
      super(TimeDistributed, self).__init__()
      self.module = module
      self.batch_first = batch_first

   def forward(self, x):

      if len(x.size()) <= 2:
         return self.module(x)
      # Squash samples and timesteps into a single axis
      x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
      y = self.module(x_reshape)
      # We have to reshape Y
      if self.batch_first:
         y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
      else:
         y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
      return y

class DCASE_Baseline(nn.Module):
   """ DCASE Baseline model
   Note: 
      - inputs size is 512 embeddings generated by openl3, 2 spatial metadata 
      and 83 time metadata (3 hot encoded metadata)
      - L2 regularization of 1e-5 (except on AutoPool)
      - Adam optimizer
      - Trained on 100 epoch max
      - Early stopping on validation loss (patience = 20)
   Args:
      input_size (int): size of the input vector
      num_classes (int): number of classes
      hidden_layer_size (int): size of hidden layer
      num_hidden_layers (int): number of hidden layers
   """

   def __init__(self, input_size, num_classes, hidden_layer_size=128, num_hidden_layers=1):
      super(DCASE_Baseline, self).__init__()
      self.layer1 = TimeDistributed(nn.Linear(input_size,hidden_layer_size))
      self.activation1 = TimeDistributed(nn.ReLU())
      self.layer2 = TimeDistributed(nn.Linear(hidden_layer_size,num_classes))
      self.activation2 = TimeDistributed(nn.Sigmoid())
      self.autopool = AutoPool(num_classes)
   
   def forward(self, x):
      x = self.layer1(x)
      x = self.activation1(x)
      x = self.layer2(x)
      x = self.activation2(x)
      x = self.autopool(x)
      return x