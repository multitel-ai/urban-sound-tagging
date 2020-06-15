#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.DCASE_baseline import AutoPool
from models.Time2vec import Time2Vec
from activation.mish import Mish
from models.TALNet_CNN14_Meta import MultiHead
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

class Normed_Linear(nn.Linear):
    """ Linear Layer with weight and input L2 normalized
    Could lead to better 'geometric' space and could deal with imbalance dataset issues
    Args:
        in_features (int) : size of each input sample
        out_features (int) : size of each output sample
        bias (bool) : If False, the layer will not learn an additive bias.
    Shape:
        Input: (N, *, in_features)
        Output: (N, *, out_features)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        weight = self.weight/(torch.norm(((self.weight)), 2, 0) + 1e-5)
        x = x/(torch.norm(((x)), 2, -1)+1e-5).unsqueeze(-1)
        return F.linear(x, weight, self.bias)

class AvgMaxPool2d(nn.Module):
    """ Average + Max Pooling layer
    Average Pooling added to Max Pooling
    Args:
        pool_stride (int, tuple) : controls the pooling stride
    """
    def __init__(self, pool_stride):
        super().__init__()
        self.pool_stride = pool_stride
        self.avgpool = nn.MaxPool2d(self.pool_stride)
        self.maxpool = nn.AvgPool2d(self.pool_stride)
    
    def forward(self, x):
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        return x1+x2

class Pooling_Head(nn.Module):
    """ Pooling layer for MIL
    Coda adapted from 'Polyphonic Sound Event Detection with Weak Labeling' Yun Wang github
    Link : https://github.com/MaigoAkisame/cmu-thesis
    Args:
        in_features (int) : size of each input sample
        out_features (int) : size of each output sample
        pooling (str) : pooling strategie, can be max, ave, lin, exp, att, auto
    """
    def __init__(self, in_features, out_features, pooling):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pooling = pooling

        if self.pooling == 'att':
            self.fc_att = nn.Linear(self.in_features, self.out_features)
            nn.init.xavier_uniform_(self.fc_att.weight); nn.init.constant_(self.fc_att.bias, 0)
        elif self.pooling == 'auto':
            self.autopool = AutoPool(self.out_features)
    
    def forward(self, frame_prob, x):
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim = 1) / frame_prob.sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim = 1) / frame_prob.exp().sum(dim = 1)
            return global_prob, frame_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x), dim = 1)
            global_prob = (frame_prob * frame_att).sum(dim = 1)
            return global_prob, frame_prob #, frame_att
        elif self.pooling == 'auto':
            global_prob = self.autopool(frame_prob)
            return global_prob, frame_prob

class ConvBlockTALNet(nn.Conv2d):
    """ TALNet ConvBlock with Weight Standardization (WS)
    Link to WS : https://arxiv.org/abs/1903.10520
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', 
                 norm=None, activation='relu', pool_stride=None, pool_strat='max'):
        # Use padding depending on kernel size by default
        if padding==None:
            padding = tuple(int(int(x)/2) for x in kernel_size)

        # Call __init__ of nn.Conv2d
        super(ConvBlockTALNet, self).__init__(in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias, padding_mode)
        
        # Initialize norm if needed (support None, Batch Norm, Group Norm)
        if norm =='GN':
            self.norm = True
            self.norm_layer = nn.GroupNorm(num_channels=self.out_channels, num_groups=32)
        elif norm =='BN':
            self.norm = True
            self.norm_layer = nn.BatchNorm2d(self.n_output)
        else:
            self.norm = False
        
        # Initialize activation function
        if activation =='relu':
            self.activation = nn.ReLU()
        elif activation =='prelu':
            self.activation = nn.PReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'mish':
            self.activation = Mish()
        else:
            raise Exception('Incorrect argument!')

        # Initialize pooling if needed (support max and avg pooling)
        self.pool_stride = pool_stride
        if pool_strat=='max':
            self.pooling = nn.MaxPool2d(self.pool_stride)
        elif pool_strat=='avg':
            self.pooling = nn.AvgPool2d(self.pool_stride)
        elif pool_strat=='avg_max':
            self.pooling = AvgMaxPool2d(self.pool_stride)
        
        # Better Initialization
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        # Z-Norm weights (Weight Standardization)
        weight = self.weight - self.weight.view(self.weight.size(0),-1,1,1).mean(1, keepdim=True)
        std = weight.view(weight.size(0),-1,1,1).std(dim=1, keepdim=True) + 1e-5 # Avoid 0 div
        weight = weight / std.expand_as(weight)

        # Compute conv2D with Z-Normed weights
        x = F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        # Apply norm if needed
        if self.norm: x = self.norm_layer(x)

        # Apply activation function
        x = self.activation(x)

        # Apply pooling if needed
        if self.pool_stride: x = self.pooling(x)
        return x

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class CNN14_meta(nn.Module):
    """ Improved TALNet architecture + metadata integration (Time2vec) + CNN14
    Coda adapted from 'Polyphonic Sound Event Detection with Weak Labeling' Yun Wang github
    Link : https://github.com/MaigoAkisame/cmu-thesis
    Args:
        num_mels (int) : number of mels of spectrogram
        num_classes (int) : number of classes
        num_meta (int) : number of metadata
        meta_emb (int) : metadata embedding size
        d_kv (int) : size of key and values of attention block
        dropout_transfo (float) : dropout applied inside attention block

    """
    def __init__(self, num_mels, num_classes, num_meta, meta_emb=64, n_head=8, d_kv=128, dropout_transfo=0, pooling='att', sample_rate=32000, window_size=1024,
        hop_size=320, mel_bins=64, fmin=50, fmax=14000):
        super(CNN14_meta, self).__init__()

        self.output_size = num_classes
        self.n_head = n_head
        self.d_k = self.d_v = d_kv
        self.dropout_transfo = dropout_transfo
        self.pooling = pooling
        # MetaData
        self.num_meta = num_meta
        self.meta_emb = meta_emb
        # CNN14
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        ###
        # META Part
        ###
        self.t2v = Time2Vec(self.num_meta, self.meta_emb)
        self.self_attention_meta = MultiHead(self.n_head, self.num_meta, self.d_k, self.d_v, self.dropout_transfo)

        ###
        # CNN14
        ###

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)

        self.multihead_CNN14 = MultiHead(self.n_head, 2048, self.d_k, self.d_v, self.dropout_transfo)

        ###
        # HEAD
        ###
        self.fc_prob = nn.Linear(2048 + self.meta_emb * self.num_meta, self.output_size)

    def forward(self, xcnn, meta):
        ###
        # CNN14 AUDIOSET
        ###
        xcnn = self.spectrogram_extractor(xcnn)   # (batch_size, 1, time_steps, freq_bins)
        xcnn = self.logmel_extractor(xcnn)    # (batch_size, 1, time_steps, mel_bins)
        
        xcnn = xcnn.transpose(1, 3)
        xcnn = self.bn0(xcnn)
        xcnn = xcnn.transpose(1, 3)
        
        if self.training:
            xcnn = self.spec_augmenter(xcnn)

        xcnn = self.conv_block1(xcnn, pool_size=(2, 2), pool_type='avg')
        xcnn = F.dropout(xcnn, p=0.2, training=self.training)
        xcnn = self.conv_block2(xcnn, pool_size=(2, 2), pool_type='avg')
        xcnn = F.dropout(xcnn, p=0.2, training=self.training)
        xcnn = self.conv_block3(xcnn, pool_size=(2, 2), pool_type='avg')
        xcnn = F.dropout(xcnn, p=0.2, training=self.training)
        xcnn = self.conv_block4(xcnn, pool_size=(2, 2), pool_type='avg')
        xcnn = F.dropout(xcnn, p=0.2, training=self.training)
        xcnn = self.conv_block5(xcnn, pool_size=(2, 2), pool_type='avg')
        xcnn = F.dropout(xcnn, p=0.2, training=self.training)
        xcnn = self.conv_block6(xcnn, pool_size=(1, 1), pool_type='avg')
        xcnn = F.dropout(xcnn, p=0.2, training=self.training)
        xcnn = torch.mean(xcnn, dim=3)
        
        (x1, _) = torch.max(xcnn, dim=2)
        x2 = torch.mean(xcnn, dim=2)
        xcnn = x1 + x2
        xcnn = F.dropout(xcnn, p=0.5, training=self.training)
        xcnn = F.relu_(self.fc1(xcnn))

        ###
        # META Part
        ###
        meta = self.t2v(meta)
        meta = self.self_attention_meta(meta, meta, meta) # [bs, n_sin, n_hid=n_meta]
        meta = meta.view((-1, meta.size(1) * meta.size(2))) # [bs, emb]

        ###
        # HEAD
        ###
        xcnn = torch.cat([xcnn, meta],1)
        xcnn = self.fc_prob(xcnn)
        xcnn = torch.sigmoid(xcnn)
        return xcnn                                                       

if __name__ == "__main__":
    # DEBUG Code
    # x = torch.randn((2,3,3,3)) #bs, c, h, w
    #x = torch.randn((4,400,64))
    x = torch.randn((4,100,64)) # bs, seq, emb
    dico = {
        'n_conv_layers':10,
        'kernel_size':(3,3),
        'n_pool_layers':5,
        'embedding_size':1024,
        'norm':'GN',
        'pooling':'att',
        'dropout':0.0,
        'conv_pool_strat':'max',
        'conv_activation':'relu',
        'n_head':8,
        'd_kv':128,
        'dropout_transfo':0,
        'num_mels':64,
        'num_classes':8,
        'num_meta':5,
        'meta_emb':64
    }
    model = TALNet(**dico)
    test = Normed_Linear(64, 8)
    #r = test(x)
    result = model(x)
# %%
