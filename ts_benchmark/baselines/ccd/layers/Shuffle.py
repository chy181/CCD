import torch
from torch import nn
from einops import rearrange
import torch.nn.init as init
           
class Shuffler(nn.Module):
    def __init__(self,n_vars, shuffle_vector_dim=4, initialization_type='kaiming'):
        super().__init__()
        self.initialization_type = initialization_type
        self.shuffle_vector_dim = shuffle_vector_dim
        self.n_vars = n_vars
        shuffle_vector_shape = tuple([self.n_vars] * shuffle_vector_dim)
        self.shuffle_vector = nn.Parameter(torch.empty(shuffle_vector_shape,device='cuda'))
        self.activation = "relu"
        self.initialize_shuffle_vector()

    def initialize_shuffle_vector(self):
        """Initialize the shuffle vector based on the user-specified initialization type."""
        if self.initialization_type == "kaiming" and self.shuffle_vector_dim > 1:
            # He initialization for higher-dimensional tensors
            init.kaiming_normal_(self.shuffle_vector, mode='fan_out', nonlinearity=self.activation)
            # Scale and shift values
            scale_factor, shift_value = 0.001, 0.01
            self.shuffle_vector.data.mul_(scale_factor).add_(shift_value)

        elif self.initialization_type == "manual" or self.shuffle_vector_dim == 1:
            # Manual initialization with user-defined scale and shift
            scale_factor, shift_value = 0.1, 0.5
            self.shuffle_vector.data.fill_(scale_factor).add_(shift_value)

    def forward(self, x ):
        # Reorganize the tensor to (b, d, p, c) format
        x = rearrange(x, 'b d c p -> b d p c')
        # Move shuffle_vector to the same device as x if needed
        if self.shuffle_vector.device != x.device:
            self.shuffle_vector = self.shuffle_vector.to(x.device)
            
        if len(self.shuffle_vector.shape) > 1:
            shuffle_vector_sum = self.shuffle_vector.sum(tuple(range(len(self.shuffle_vector.shape) - 1)))
        else:
            shuffle_vector_sum = self.shuffle_vector
        self.descending_indices = torch.argsort(shuffle_vector_sum, descending=True)

        # [batch_size, n_vars, patch_num, 1]
        shuffled_scores = torch.gather(input=shuffle_vector_sum, index=self.descending_indices, dim=0)
        non_zero_mask = shuffled_scores != 0
        inv = (1 / shuffled_scores[non_zero_mask] + 1e-5).detach()

        # [batch_size, n_vars, patch_num, patch_size]
        shuffle_channel_indices = self.descending_indices.repeat(x.size(0), x.size(1), x.size(2), 1)
        shuffled_channels = torch.gather(input=x, index=shuffle_channel_indices, dim=3)
        shuffled_scores[non_zero_mask] *= inv
        x = shuffled_channels
        x = rearrange(x, 'b d p c -> b d c p')
        self.shuffle_vector_sum = shuffle_vector_sum
        return x
  
    def invert(self,x):
        indices = torch.argsort(self.descending_indices)
        indices = indices.repeat(x.size(0), x.size(1), x.size(3), 1).permute(0,1,3,2)
        x = torch.gather(input=x, index=indices, dim=2)
        return x