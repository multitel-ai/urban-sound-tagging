import torch
import torch.nn as nn

class Time2Vec(nn.Module):
   """ Time2Vec
   Inspired of : https://github.com/ojus1/Time2Vec-PyTorch
   and https://discuss.pytorch.org/t/how-to-get-the-batch-dimension-right-in-the-forward-path-of-a-custom-layer/80131/2
   Original paper : https://arxiv.org/pdf/1907.05321.pdf
   Keras implementation : https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
   """

   def __init__(self, input_dim, output_dim):
      super().__init__()
      self.output_dim = output_dim
      self.input_dim = input_dim
      self.w0 = nn.Parameter(torch.Tensor(1, input_dim))
      self.phi0 = nn.Parameter(torch.Tensor(1, input_dim))
      self.W = nn.Parameter(torch.Tensor(input_dim, output_dim-1))
      self.Phi = nn.Parameter(torch.Tensor(input_dim, output_dim-1))
      self.reset_parameters()

   def reset_parameters(self):
      nn.init.uniform_(self.w0, 0, 1)
      nn.init.uniform_(self.phi0, 0, 1)
      nn.init.uniform_(self.W, 0, 1)
      nn.init.uniform_(self.Phi, 0, 1)

   def forward(self, x):
      n_batch = x.size(0)
      original = (x*self.w0 + self.phi0).unsqueeze(-1)
      x = torch.repeat_interleave(x, repeats=self.output_dim-1, dim=0).view(n_batch,-1,self.output_dim-1)
      x = torch.sin(x * self.W + self.Phi)
      return torch.cat([original,x],-1).view(n_batch,self.output_dim,-1).contiguous()