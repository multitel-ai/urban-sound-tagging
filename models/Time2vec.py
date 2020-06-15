#%%
import torch
import torch.nn as nn

# Inspired of : https://github.com/ojus1/Time2Vec-PyTorch
# TLSTM3 ? : https://github.com/ZJULearning/time_lstm

class TV2(nn.Module):
   def __init__(self, in_features, n_sin):
      super(TV2, self).__init__()
      self.n_sin = n_sin
      self.w0 = nn.Parameter(torch.randn(in_features, 1))
      self.b0 = nn.Parameter(torch.randn(in_features, 1))
      self.w = nn.Parameter(torch.randn(in_features, n_sin))
      self.b = nn.Parameter(torch.randn(in_features, n_sin))
      self.f = torch.sin
   
   def t2v(self, tau):
      print(tau)
      v1 = self.w0 * tau + self.b0
      print(v1)
      t1 = tau.repeat(1, self.n_sin)
      print(t1.size(), t1)
      print(self.w)
      v2 = self.f(torch.mm(t1, torch.t(self.w)) + self.b)
      return torch.cat([v1, v2], 1)

   def forward(self, tau):
      return self.t2v(tau)

class T2VLSTM(nn.Module):
   def __init__(self, num_sin, out_dim):
      super(T2VLSTM, self).__init__()

      self.l1 = TV2(1, num_sin)
      # output hidden_dim+1
      self.lstm = nn.LSTM(num_sin+1, out_dim)
      #self.fc1 = nn.Linear(num_sin+1, 2)
    
   def forward(self, x):
      x = self.l1(x)
      x = self.lstm(x)
      return x

class T2VClassifier1(nn.Module):
   def __init__(self, num_classes, num_sin=64, out_dim=128):
      super(T2VClassifier1, self).__init__()

      self.t2v = TV2(1, num_sin)
      self.l1 = nn.Linear(num_sin+1, num_classes)
      self.activation1 = nn.Sigmoid()
      
   def forward(self, x):
      x = self.t2v(x)
      x = self.l1(x)
      x = self.activation1(x)
      return x

class T2VClassifier2(nn.Module):
   def __init__(self, num_classes, input_dim=3, num_sin=64):
      super(T2VClassifier2, self).__init__()
      self.l1 = nn.Linear(input_dim,1)
      self.t2v = TV2(1, num_sin)
      self.l2 = nn.Linear(num_sin+1, num_classes)
      self.activation1 = nn.Sigmoid()
      
   def forward(self, x):
      x = self.l1(x)
      x = self.t2v(x)
      x = self.l2(x)
      x = self.activation1(x)
      return x

class T2VClassifier3(nn.Module):
   """ Test for 2 inputs only 
   """
   def __init__(self, num_classes, num_sin=64):
      super(T2VClassifier3, self).__init__()
      self.l1L = nn.Linear(1,1)
      self.l1l = nn.Linear(1,1)
      self.t2vL = TV2(1, num_sin)
      self.t2vl = TV2(1, num_sin)
      self.l2 = nn.Linear((num_sin+1)*2, num_classes)
      self.activation1 = nn.Sigmoid()
      
   def forward(self, x):
      xL, xl = torch.split(x, 1, dim=1)
      xL, xl = self.l1L(xL), self.l1l(xl)
      xL, xl = self.t2vL(xL), self.t2vl(xl)
      x = torch.cat((xL,xl), 1)
      x = self.l2(x)
      x = self.activation1(x)
      return x

class Time2Vec(nn.Module):

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

"""
source: https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e
and:    https://arxiv.org/pdf/1907.05321.pdf
https://discuss.pytorch.org/t/how-to-get-the-batch-dimension-right-in-the-forward-path-of-a-custom-layer/80131/2
"""

if __name__ == '__main__':
   n_batch = 2
   n_input = 3
   n_sin = 10
   x = torch.randn((n_batch, n_input)) # (batch_s, n_temps_to_encode)
   lstm = nn.LSTM(n_input, 64)
   #model = TV2(3, 4)
   model = Time2Vec(n_input,n_sin)
   x = model(x)
   print(x.size())
   x, _ = lstm(x)
   x = x[:,-1,:]
   print(x.size())
   print(x)

# %%


# %%
