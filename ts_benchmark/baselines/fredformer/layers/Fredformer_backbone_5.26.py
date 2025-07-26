__all__ = ['FT_backbone']

# Cell
import torch
from torch import nn
import torch.nn.functional as F
from ..layers.RevIN import RevIN
from ..layers.cross_Transformer_nys import Trans_C as Trans_C_nys
from ..layers.cross_Transformer import Trans_C
# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class Fredformer_backbone(nn.Module):
    def __init__(self, ablation:int,  mlp_drop:float, use_nys:int, output:int, mlp_hidden:int,cf_dim:int,cf_depth :int,cf_heads:int,cf_mlp:int,cf_head_dim:int,cf_drop:float,c_in:int, context_window:int, target_window:int, patch_len:int, stride:int,  d_model:int, 
                head_dropout = 0, padding_patch = None,individual = False, revin = True, affine = True, subtract_last = False, **kwargs):
        
        super().__init__()
        self.use_nys = use_nys
        self.ablation = ablation
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.output = output
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.targetwindow=target_window
        self.horizon = self.targetwindow
        patch_num = int((context_window - patch_len)/stride + 1)
        self.norm = nn.LayerNorm(patch_len)
        #print("depth=",cf_depth)
        # Backbone 
        self.re_attn = True
        if self.use_nys==0:
            self.fre_transformer = Trans_C(dim = cf_dim,depth = cf_depth, heads = cf_heads, mlp_dim = cf_mlp, dim_head = cf_head_dim, dropout = cf_drop, patch_dim = patch_len*2 , horizon = self.horizon*2, d_model=d_model*2)
            self.fre_transformer1 = Trans_C(dim = cf_dim,depth = cf_depth, heads = cf_heads, mlp_dim = cf_mlp, dim_head = cf_head_dim, dropout = cf_drop, patch_dim = patch_len*2 , horizon = self.horizon*2, d_model=d_model*2)
        else:
            self.fre_transformer = Trans_C_nys(dim = cf_dim,depth = cf_depth, heads = cf_heads, mlp_dim = cf_mlp, dim_head = cf_head_dim, dropout = cf_drop, patch_dim = patch_len*2 , horizon = self.horizon*2, d_model=d_model*2)
        
        
        # Head
        self.head_nf_f  = d_model * 2 * patch_num #self.horizon * patch_num#patch_len * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=head_dropout)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=head_dropout)
        self.head_f3 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=head_dropout)
        self.head_f4 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=head_dropout)
        
        self.context_window = context_window
        self.ircom = nn.Linear(self.targetwindow*2,self.targetwindow)
        self.ircom1 = Head(self.targetwindow*2,self.targetwindow*2)
        self.ircom2 = nn.Linear(self.targetwindow*4,self.targetwindow*2)
        self.in_layer_r = nn.Linear(self.context_window,self.context_window)
        self.in_layer_i = nn.Linear(self.context_window,self.context_window)
        #break up R&I:s
        self.get_r = nn.Linear(d_model*2,d_model*2)
        self.get_i = nn.Linear(d_model*2,d_model*2)
        self.get_r1 = nn.Linear(d_model*2,d_model*2)
        self.get_i1 = nn.Linear(d_model*2,d_model*2)

        self.imag_encoder = nn.Linear(context_window,self.targetwindow)
        self.real_encoder = nn.Linear(context_window,self.targetwindow)

        self.imag_encoder1 = nn.Linear(context_window,self.targetwindow)
        self.real_encoder1 = nn.Linear(context_window,self.targetwindow)
        self.decompose = series_decomp(25)

    def frequency_transformer(self,z):
        z1 = z.real
        z2 = z.imag
        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z2: [bs x nvars x patch_num x patch_len]                                                                 
        #for channel-wise_1
        z1 = z1.permute(0,2,1,3)
        z2 = z2.permute(0,2,1,3)

        # model shape
        batch_size = z1.shape[0]
        patch_num  = z1.shape[1]
        c_in       = z1.shape[2]
        patch_len  = z1.shape[3]
        
        #proposed
        z1 = torch.reshape(z1, (batch_size*patch_num,c_in,z1.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size*patch_num,c_in,z2.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]

        z = self.fre_transformer(torch.cat((z1,z2),-1))
        
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size,patch_num,c_in,z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size,patch_num,c_in,z2.shape[-1]))
        z1 = z1.permute(0,2,1,3)                                                                    # z1: [bs, nvars， patch_num, horizon]
        z2 = z2.permute(0,2,1,3)
        return self.head_f1(z1),self.head_f2(z2)
    
    def quefrency_transformer(self,z):
        z1 = z.real
        z2 = z.imag
        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z2: [bs x nvars x patch_num x patch_len]                                                                 
        #for channel-wise_1
        z1 = z1.permute(0,2,1,3)
        z2 = z2.permute(0,2,1,3)

        # model shape
        batch_size = z1.shape[0]
        patch_num  = z1.shape[1]
        c_in       = z1.shape[2]
        patch_len  = z1.shape[3]
        
        #proposed
        z1 = torch.reshape(z1, (batch_size*patch_num,c_in,z1.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size*patch_num,c_in,z2.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]

        z = self.fre_transformer1(torch.cat((z1,z2),-1))
        
        z1 = self.get_r1(z)
        z2 = self.get_i1(z)

        z1 = torch.reshape(z1, (batch_size,patch_num,c_in,z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size,patch_num,c_in,z2.shape[-1]))
        z1 = z1.permute(0,2,1,3)                                                                    # z1: [bs, nvars， patch_num, horizon]
        z2 = z2.permute(0,2,1,3)
        return self.head_f3(z1),self.head_f4(z2)


    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]

        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        z = torch.fft.fft(z)     
        z1f,z2f = self.frequency_transformer(z)
        
        frequency = torch.complex(z1f,z2f)
        
        z = torch.log(frequency + 1e-8)
        z = torch.fft.ifft(z)
        z1,z2 = self.quefrency_transformer(z)
        quefrency = torch.complex(z1,z2)

        z = torch.fft.fft(quefrency)
        z = torch.exp(z)-1e-8
        # z = (z + frequency)/2
        # z = frequency
        z = self.ircom2(torch.cat((z.real,z.imag,frequency.real,frequency.imag),-1))
        z = torch.complex(z[...,:self.targetwindow],z[...,self.targetwindow:])
        z = torch.fft.ifft(z)
        # z = z.real 
        # zi = z.imag
        z = self.ircom(torch.cat((z.real,z.imag),-1))

        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
    
        return z,frequency,quefrency #quefrency

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears1 = nn.ModuleList()
            #self.linears2 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, target_window))
                #self.linears2.append(nn.Linear(target_window, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears1[i](z)                    # z: [bs x target_window]
                #z = self.linears2[i](z)                    # z: [target_window x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
            #x = self.linear1(x)
            #x = self.linear2(x) + x
            #x = self.dropout(x)
        return x
class Head(nn.Module):
    def __init__(self, dim_in, dim_out, dim=768, dropout=0.3):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_in, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim_out)
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        x = self.dropout(x)
        x = F.selu(self.linear1(x))
        # x = F.relu(self.linear2(x)) + x
        x = self.linear3(x)
        return x
    
class Flatten_Head_t(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout):
        super().__init__()
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, nf)
        self.linear2 = nn.Linear(nf, nf)
        self.linear3 = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        
        x = self.flatten(x)
        x = F.relu(self.linear1(x)) + x
        x = F.relu(self.linear2(x)) + x
        
        x = self.linear3(x)
        return x