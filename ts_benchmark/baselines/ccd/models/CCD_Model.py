from torch import nn
from ..layers.RevIN import RevIN
from ..layers.views import CFView,CSView

class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.configs = configs
        
        self.cs_view = CSView(layers=configs.layers, cf_dim=configs.cf_dim, seq_len=configs.seq_len, pred_len=configs.pred_len, d_model=configs.series_dim, enc_in=configs.enc_in, dropout=configs.dropout, head_dropout=configs.head_dropout, s_size=configs.s_size, v_size=configs.v_size, sample_rate=configs.sample_rate, max_number=configs.max_number,s_shuffle_dim=configs.s_shuffle_dim,c_shuffle_dim=configs.c_shuffle_dim,configs=configs)
        
        self.cf_view = CFView(layers=configs.layers, cf_dim=configs.cf_dim, seq_len=configs.seq_len, pred_len=configs.pred_len, d_model=configs.d_model, enc_in=configs.enc_in, dropout=configs.dropout, head_dropout=configs.head_dropout, f_size=configs.f_size, v_size=configs.v_size, patch_len=configs.patch_len, max_number=configs.max_number,f_shuffle_dim=configs.f_shuffle_dim,c_shuffle_dim=configs.c_shuffle_dim,configs=configs)

    
    def forward(self, x):        # x: [Batch, Input length, Channel]
        # x: [Batch, Channel, Input length]
        if self.revin: 
            x = self.revin_layer(x, 'norm')

        x = x.permute(0,2,1)
        output = self.cs_view(x) + self.cf_view(x) 
        output = output.permute(0,2,1)
        
        # denorm
        if self.revin: 
            output = self.revin_layer(output, 'denorm')

        return output