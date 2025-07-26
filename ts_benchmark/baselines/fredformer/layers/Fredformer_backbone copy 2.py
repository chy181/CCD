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
        self.context_window = context_window
        self.ircom = nn.Linear(self.targetwindow*2,self.targetwindow)
        self.ircom1 = nn.Linear(context_window*2,context_window*2)
        # nn.Sequential(nn.Linear(self.targetwindow*2,512),
        #                             nn.ReLU(),
        #                             nn.Dropout(0.2),
        #                             # nn.Linear(512,512),
        #                             # nn.ReLU(),
        #                             nn.Linear(512,self.targetwindow*2))

        self.rfftlayer = nn.Linear(self.targetwindow*2-2,self.targetwindow)
        self.final = nn.Linear(self.targetwindow*2,self.targetwindow)

        #break up R&I:
        self.get_r = nn.Linear(d_model*2,d_model*2)
        self.get_i = nn.Linear(d_model*2,d_model*2)
        self.get_r1 = nn.Linear(d_model*2,d_model*2)
        self.get_i1 = nn.Linear(d_model*2,d_model*2)

        self.output1 = nn.Linear(target_window,target_window)


        #ablation
        self.input = nn.Linear(c_in,patch_len*2)
        self.outpt = nn.Linear(d_model*2,c_in)
        self.abfinal = nn.Linear(patch_len*patch_num,target_window)
    
    def transform(self,z):
        z = torch.fft.fft(z)
        z = torch.log(z + 1e-10)
        z = torch.fft.ifft(z)
        return z
    
    def itransform(self,z1,z2):
        
        z1 = self.head_f1(z1)                                                                    # z: [bs x nvars x target_window] 
        z2 = self.head_f2(z2)

        z = torch.complex(z1,z2)
        z = torch.fft.ifft(z)
         
        return z
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]

        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
        
        x = self.transform(z)
        z = torch.fft.fft(z)
        
        z1 = z.real
        z2 = z.imag
        x1 = x.real
        x2 = x.imag
        

        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z2: [bs x nvars x patch_num x patch_len]                                                                 
        x1 = x1.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z1: [bs x nvars x patch_num x patch_len]
        x2 = x2.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z2: [bs x nvars x patch_num x patch_len]    

        #for channel-wise_1
        z1 = z1.permute(0,2,1,3)
        z2 = z2.permute(0,2,1,3)
        x1 = x1.permute(0,2,1,3)
        x2 = x2.permute(0,2,1,3)


        # model shape
        batch_size = z1.shape[0]
        patch_num  = z1.shape[1]
        c_in       = z1.shape[2]
        patch_len  = z1.shape[3]
        
        #proposed
        z1 = torch.reshape(z1, (batch_size*patch_num,c_in,z1.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size*patch_num,c_in,z2.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]

        x1 = torch.reshape(x1, (batch_size*patch_num,c_in,z1.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]
        x2 = torch.reshape(x2, (batch_size*patch_num,c_in,z2.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]

        x =  self.fre_transformer1(torch.cat((x1,x2),-1))
        z = self.fre_transformer(torch.cat((z1,z2),-1))
        
        z1 = self.get_r(z) + self.get_r1(x)
        z2 = self.get_i(z) + self.get_i1(x)
        z1 = torch.reshape(z1, (batch_size,patch_num,c_in,z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size,patch_num,c_in,z2.shape[-1]))
        z1 = z1.permute(0,2,1,3)                                                                    # z1: [bs, nvars， patch_num, horizon]
        z2 = z2.permute(0,2,1,3)
        z = self.itransform(z1,z2)
        zr = z.real                                              
        zi = z.imag
        z = self.ircom(torch.cat((zr,zi),-1))


        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z

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