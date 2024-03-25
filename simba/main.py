import torch 
from torch import nn, Tensor
from zeta.nn import MambaBlock


class EMMImage(nn.Module):
    """
    EMM (Element-wise Multiplication Module) is a PyTorch module that performs element-wise multiplication
    between two tensors.

    Args:
        None

    Returns:
        Tensor: The result of element-wise multiplication between the input tensor `x` and the weight tensor.

    """

    def __init__(
        self,
    ):
        super().__init__()
    
    def forward(self, x: Tensor, weight: Tensor) -> Tensor:
        x_b, x_h, x_w, x_c = x.shape
        
        # Weight shape
        c_b, c_d, c_d = weight.shape
        
        # Something
        
        # Multiply
        return x * weight
        

class EinFFT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        heads: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.heads = heads
        
        # FFT
        self.fft = torch.fft.fft()
        
        # Weights for Wr and Wi
        
    def forward(self, x: Tensor):
        x = self.fft(x)
        
        # Get Xr and X1
        fast_fouried = torch.fft.fft(x)
        
        # Get Wr Wi
        xr = fast_fouried
        xi = fast_fouried
        
        # EMM
        EMM()(xr, xi)
        
        

class Simba(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        heads: int = 8,
        dim_head: int = 64,
        d_state: int = 64,
        d_conv: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.d_state = d_state
        self.d_conv = d_conv
        self.dropout = nn.Dropout(dropout)
        
        
        # Mamba Block
        self.mamba = MambaBlock(
            dim = self.dim,
            depth = 1,
            d_state = self.d_state,
            d_conv = self.d_conv,
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.shape
        
        residual = x
        
        # Layernorm
        normed = nn.LayerNorm(d)(x)
        
        # Mamba
        mamba = self.mamba(normed)
        
        # Dropout
        droped = self.dropout(mamba)
        
        out = residual + droped
        
        # Phase 2
        residual_new = out
        
        # Layernorm
        normed_new = nn.LayerNorm(d)(out)
        
        # einfft
        fasted = normed_new
        
        # Dropout
        out = self.dropout(fasted)
        
        # residual
        return out + residual_new