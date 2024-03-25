import torch
from torch import nn, Tensor
from zeta.nn import MambaBlock


class EinFFT(nn.Module):
    """
    EinFFT module performs the EinFFT operation on the input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dim (int): Dimension of the input tensor.
        heads (int, optional): Number of attention heads. Defaults to 8.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dim (int): Dimension of the input tensor.
        heads (int): Number of attention heads.
        act (nn.SiLU): Activation function (SiLU).
        Wr (nn.Parameter): Learnable weight parameter for real part.
        Wi (nn.Parameter): Learnable weight parameter for imaginary part.

    """

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

        # silu
        self.act = nn.SiLU()

        # Weights for Wr and Wi
        self.Wr = nn.Parameter(torch.randn(in_channels, out_channels))
        self.Wi = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the EinFFT module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, in_channels, height, width).

        """
        b, c, h, w = x.shape

        # Get Xr and X1
        fast_fouried = torch.fft.fft(x)
        print(fast_fouried.shape)

        # Get Wr Wi use pytorch split instead
        xr = fast_fouried.real
        xi = fast_fouried.imag

        # Einstein Matrix Multiplication with XR, Xi, Wr, Wi use torch split instead
        # matmul = torch.matmul(xr, self.Wr) + torch.matmul(xi, self.Wi)
        matmul = torch.matmul(xr, xi)
        # matmul = torch.matmul(self.Wr, self.Wi)
        print(matmul.shape)

        # Xr, Xi hat, use torch split instead
        xr_hat = matmul  # .real
        xi_hat = matmul  # .imag

        # Silu
        acted_xr_hat = self.act(xr_hat)
        acted_xi_hat = self.act(xi_hat)

        # Emm with the weights use torch split instead
        # emmed = torch.matmul(
        #     acted_xr_hat,
        #     self.Wr
        # ) + torch.matmul(
        #     acted_xi_hat,
        #     self.Wi
        # )
        emmed = torch.matmul(acted_xr_hat, acted_xi_hat)

        # Split up into Xr and Xi again for the ifft use torch split instead
        xr_hat = emmed  # .real
        xi_hat = emmed  # .imag

        # IFFT
        iffted = torch.fft.ifft(xr_hat + xi_hat)

        return iffted


x = torch.randn(1, 3, 64, 64)
einfft = EinFFT(3, 64, 64)
out = einfft(x)
print(out)


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
