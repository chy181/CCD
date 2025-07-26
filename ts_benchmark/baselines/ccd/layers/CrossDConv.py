
import torch
from torch import nn
import math
from einops import rearrange
import torch.nn.functional as F

class CrossDConv(nn.Module):
    def __init__(self, inc, outc, t_size, c_size, stride=1, sigma=2.0, max_number=10, bias=None, modulation=True,head_dims=0,configs=None):
        super(CrossDConv, self).__init__()
    
        # --- Parameters (mostly unchanged) ---
        self.t_size = t_size
        self.sigma = sigma
        self.max_number = max_number
        self.c_size = c_size
        self.modulation = modulation
        self.stride = stride
        self.configs =  configs
        self.register_buffer("p_n", self._get_p_n(N=t_size + c_size, dtype=torch.float32))

        # --- Convolution Layers (unchanged) ---
        self.pointwise_conv = nn.Conv2d(inc, outc, kernel_size=1, bias=True)
        
       
        if self.f_number > 0:
            self.p_conv_f = nn.Sequential(
                nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride, groups=inc),
                nn.Conv2d(inc, 2 * self.f_number, kernel_size=1)
            )
        self.p_conv_f.register_full_backward_hook(self._set_lr)
        if self.c_number > 0:
            self.p_conv_c = nn.Sequential(
                nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride, groups=inc),
                nn.Conv2d(inc, 2 * self.c_number, kernel_size=1)
            )
        self.p_conv_c.register_full_backward_hook(self._set_lr)

        if self.modulation:
            self.m_conv = nn.Sequential(
                nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride, groups=inc),
                nn.Conv2d(inc, self.all_numbers + 1, kernel_size=1)
            )
            nn.init.constant_(self.m_conv[1].weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

        self.mlp = nn.Sequential(nn.Linear(outc, outc), nn.ReLU(), nn.Linear(outc, outc))
        self.norm = nn.LayerNorm(outc)
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = tuple(g * 0.1 for g in grad_input)
        grad_output = tuple(g * 0.1 for g in grad_output)
        pass

    def forward(self, x):
        b, c, h_in, w_in = x.shape

        # 1. PREDICT OFFSETS AND MODULATION MASK (Unchanged)
        offset_f = self.p_conv_f(x) if hasattr(self, 'p_conv_f') else torch.empty(b, 0, h_in, w_in, device=x.device)
        offset_c = self.p_conv_c(x) if hasattr(self, 'p_conv_c') else torch.empty(b, 0, h_in, w_in, device=x.device)
        offset = torch.cat((offset_f, offset_c), dim=1)


        h_out, w_out = offset.size(2), offset.size(3)

        modulation_scores = self.m_conv(x)
        sparsity_ratio = modulation_scores[:,-1].unsqueeze(1)
        modulation_scores = modulation_scores[:,:self.all_numbers]

        tau = 0.1
        soft_mask = torch.sigmoid((modulation_scores - sparsity_ratio) / tau)
        masked_scores = modulation_scores + torch.log(soft_mask + 1e-10)
        final_modulation_mask = torch.softmax(masked_scores, dim=1)

       
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h_out * self.stride, self.stride, device=x.device, dtype=x.dtype), torch.arange(0, w_out * self.stride, self.stride, device=x.device, dtype=x.dtype), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        p_n = self.p_n.view(1, 2, self.all_numbers, 1, 1).permute(0, 2, 3, 4, 1)
        offset = offset.view(b, 2, self.all_numbers, h_out, w_out).permute(0, 2, 3, 4, 1)

        absolute_coords = grid + p_n + offset
        norm_coords = absolute_coords.clone()
        norm_coords[..., 0] = 2.0 * absolute_coords[..., 0] / (w_in - 1) - 1.0
        norm_coords[..., 1] = 2.0 * absolute_coords[..., 1] / (h_in - 1) - 1.0
        norm_coords =norm_coords.reshape(b, self.all_numbers * h_out, w_out, 2)
        
        sampled_features = F.grid_sample(x.detach(),  norm_coords, mode='bilinear', padding_mode='zeros', align_corners=True) # Changed to bilinear
       
        sampled_features = sampled_features.view(b, c, self.all_numbers, h_out, w_out)

        
        modulated_features = sampled_features * final_modulation_mask.unsqueeze(1)

        # Aggregate the features from the selected points.
        aggregated_features = torch.sum(modulated_features, dim=2)

        # 6. APPLY FINAL CONV and MLP (MODIFIED)
        out = self.pointwise_conv(aggregated_features)
        
        out_permuted = rearrange(out, 'b d h w -> b (h w) d')
        out_mlp = self.mlp(out_permuted)
        out = rearrange(out_mlp, 'b (h w) d -> b d h w', h=h_out, w=w_out)
        return  out + x
        
    def gaussian_decay(self, length, max_val, sigma):
        # Corrected torch.range to torch.arange
        x = torch.arange(length, dtype=torch.float32)
        y = max_val * torch.exp(-(x**2) / (2 * sigma**2))
        return (y // 1 + 1).int()

    def _get_p_n(self, N, dtype):

        f_number = self.gaussian_decay(self.t_size,self.max_number,self.sigma)
        c_number = self.gaussian_decay(self.c_size,self.max_number,self.sigma)
        # Your _get_p_n logic...   
        p_n_x = []
        p_n_y = []

        if self.t_size>0:
            for i in range(1,self.t_size):
                for _ in range(f_number[i].item()):
                    p_n_x.append(0)
                    p_n_y.append(i)

        if self.c_size>0:
            for i in range(self.c_size):
                for _ in range(c_number[i]):
                    p_n_x.append(i)
                    p_n_y.append(0)
        if self.t_size>0:
            self.f_number = sum(f_number).item()-self.max_number-1
        else: 
            self.f_number = 0
            
        s = self.c_size
        for i in range(s):
            for j in range(s):
                p_n_x.append(i)
                p_n_y.append(j)
        p_n_x = torch.tensor(p_n_x, dtype=dtype)
        p_n_y = torch.tensor(p_n_y, dtype=dtype)
        p_n = torch.cat([p_n_x, p_n_y], 0)
        self.all_numbers = p_n_x.shape[0]
        p_n = p_n.view(1, 2 * self.all_numbers, 1, 1).type(dtype)
        return p_n