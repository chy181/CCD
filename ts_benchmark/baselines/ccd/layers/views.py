from torch import nn
from .CrossDConv import CrossDConv
from .Shuffle import Shuffler
from einops import rearrange
import torch
import torch.nn.functional as F
import math
class Flatten_MLP_Head(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, nf)
        self.linear2 = nn.Linear(nf, nf)
        self.linear3 = nn.Linear(nf, nf)
        self.linear4 = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.linear1(x)) + x
        x = F.relu(self.linear2(x)) + x
        x = F.relu(self.linear3(x)) + x
        x = self.linear4(x)
        return x

class Flatten_Linear_Head(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear2 = nn.Linear(nf, target_window)
        # self.linear1 = nn.Linear(nf, nf)
        self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        # x = self.flatten(x)
        x = self.dropout(x)
        # x = F.relu(self.linear1(x)) + x
        x = self.linear2(x)
        return x
    
class CFView(nn.Module):
    def __init__(self, layers, cf_dim, seq_len, pred_len, d_model, enc_in, dropout, head_dropout, f_size, v_size, patch_len, sigma=1.5, max_number=3, c_shuffle_dim=1,f_shuffle_dim=2,configs=None):
        
        super().__init__()
        self.layers = layers
        dim = cf_dim
        num_channel = enc_in
        num_frequency = (seq_len - patch_len) // patch_len + 1
        self.patch_len = patch_len
        self.dropout = nn.Dropout(dropout)
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_len*2, dim),nn.Dropout(dropout))

        self.dconvs = nn.ModuleList(CrossDConv(dim, dim, f_size, v_size, sigma=sigma, max_number=max_number,configs=configs) for _ in range(self.layers))
        if configs.ablation != 3:
            self.channel_shuffler = nn.ModuleList(Shuffler(num_channel,shuffle_vector_dim=c_shuffle_dim) for _ in range(1))
            self.frequency_shuffler = nn.ModuleList(Shuffler(num_frequency,shuffle_vector_dim=f_shuffle_dim) for _ in range(1))
        self.configs = configs
        self.mlp_head = nn.Linear(dim, d_model*2)
        self.get_r = nn.Linear(d_model*2,d_model*2)
        self.get_i = nn.Linear(d_model*2,d_model*2)
        self.head_f1 = Flatten_MLP_Head(d_model * 2 * num_frequency , pred_len, head_dropout=head_dropout)
        self.head_f2 = Flatten_MLP_Head(d_model * 2 * num_frequency , pred_len, head_dropout=head_dropout)
        self.ircom = nn.Linear(pred_len*2, pred_len)

    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag
        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                         # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)                         # z2: [bs x nvars x patch_num x patch_len]                                                                 
        #for channel-wise_1
        z1 = z1.permute(0,2,1,3)
        z2 = z2.permute(0,2,1,3)
        x = torch.cat((z1,z2),-1)
        x = self.to_patch_embedding(x)
        x = rearrange(x,'b f c d -> b d f c')
        if self.configs.ablation != 3:
            x = self.frequency_shuffler[0](x)
        x = rearrange(x,'b d f c -> b d c f')
        if self.configs.ablation != 3:
            x = self.channel_shuffler[0](x)

        for i in range(self.layers):
            x = self.dconvs[i](x) #+ x
            pass
        if self.configs.ablation != 3:
            x = self.channel_shuffler[0].invert(x)
        x = rearrange(x,'b d c f -> b d f c')
        if self.configs.ablation != 3:
            x = self.frequency_shuffler[0].invert(x)

        x = rearrange(x,'b d f c -> b f c d')
        z = self.mlp_head(x)

        z1 = self.get_r(z)
        z2 = self.get_i(z) 
        z1 = z1.permute(0,2,1,3)                                                                    # z1: [bs, nvars， patch_num, horizon]
        z2 = z2.permute(0,2,1,3)
        z1f = self.head_f1(z1)
        z2f = self.head_f2(z2)
        frequency = torch.complex(z1f,z2f)
        z = torch.fft.ifft(frequency)
        return self.ircom(torch.cat((z.real,z.imag),-1))



class CSView(nn.Module):
    def __init__(self, layers, cf_dim, seq_len, pred_len, d_model, enc_in, dropout, head_dropout,  s_size, v_size, sample_rate=5, sigma=1.5, max_number=3, c_shuffle_dim=1,s_shuffle_dim=2,configs=None):
        
        super().__init__()

        self.layers = layers
        dim = d_model
        num_channel = enc_in
        
        factors = set()  # 使用集合来自动处理重复的因数
        for i in range(1, int(math.sqrt(seq_len)) + 1):
            if seq_len % i == 0:
                factors.add(i)
                factors.add(seq_len // i)
        sample_rate= sorted(list(factors))[:sample_rate]
        # sample_rate = [2**i for i in range(sample_rate)]

        # sample_rate = [i+1 for i in range(sample_rate)]
        num_scale = len(sample_rate)
   
        self.dropout = nn.Dropout(dropout)

        self.dconvs = nn.ModuleList(CrossDConv(dim, dim, s_size, v_size, sigma=sigma,max_number=max_number,configs=configs) for _ in range(self.layers))
        
        if configs.ablation != 3:
            self.channel_shuffler = nn.ModuleList(Shuffler(num_channel ,shuffle_vector_dim=c_shuffle_dim) for _ in range(1))
            self.scale_shuffler = nn.ModuleList(Shuffler(num_scale,shuffle_vector_dim=s_shuffle_dim) for _ in range(1))
        self.configs = configs

        self.mlp_head = Flatten_MLP_Head(d_model* num_scale, pred_len, head_dropout=head_dropout)
        
        self.conv_downsampler = nn.ModuleList(torch.nn.Conv1d(
            in_channels=num_channel, 
            out_channels=num_channel, kernel_size=i, stride=i, groups=num_channel
        ) for i in sample_rate)
        self.multi_embeddings = nn.ModuleList(
           nn.Sequential(nn.Linear(seq_len//i, d_model)) for i in sample_rate)      
        

    def forward(self, z):      

        x=[]
        for i, down_conv in enumerate(self.conv_downsampler):
            sample = down_conv(z)
            x.append(self.multi_embeddings[i](sample))
        # b s c d
        x = torch.stack(x,dim=1)
        x = rearrange(x,'b s c d -> b d s c')

        if self.configs.ablation != 3:
            x = self.scale_shuffler[0](x)
        x = rearrange(x,'b d s c -> b d c s')
        if self.configs.ablation != 3:
            x = self.channel_shuffler[0](x)

        for i in range(self.layers):
            x = self.dconvs[i](x) #+ x
            pass

        if self.configs.ablation != 3:        
            x = self.channel_shuffler[0].invert(x)
        x = rearrange(x,'b d c s -> b d s c')
        if self.configs.ablation != 3:
            x = self.scale_shuffler[0].invert(x)

        x = rearrange(x,'b d s c -> b c s d')
        
        z = self.mlp_head(x)
        return z