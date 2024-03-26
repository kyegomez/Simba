import torch
from torch import nn, Tensor
from zeta.nn import MambaBlock
from simba_torch.ein_fft import EinFFT

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
            dim=self.dim,
            depth=1,
            d_state=self.d_state,
            d_conv=self.d_conv,
        )
        
        # EinFFT
        self.einfft = EinFFT(
            in_channels=self.dim,
            out_channels=self.dim,
            dim=self.dim,
        )
        

    def forward(self, x: Tensor) -> Tensor:
        #patch
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
        fasted = self.einfft(normed_new)

        # Dropout
        out = self.dropout(fasted)

        # residual
        return out + residual_new


# Forward pass with images
img = torch.randn(1, 3, 224, 224)

# Create model
model = Simba(
    dim = 3,
    dropout = 0.1,
    heads=8,
    dim_head=64,
    d_state=64,
    d_conv=64,
)

# out
out = model(img)

# Print output shape
print(out.shape)