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

        # complex weights for channel-wise transformation
        self.complex_weight = nn.Parameter(
            torch.randn(
                in_channels, out_channels, dtype=torch.complex64
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the EinFFT module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, in_channels, height, width).

        """
        b, c, h, w = x.shape

        # apply 2D FFTSolver, transform input tensor to frequency domain
        fast_fouried = torch.fft.fft2(x, dim=(-2, -1))

        # complex-valued multiplication
        einsum_mul = torch.einsum(
            "bchw,cf->bhwf", fast_fouried, self.complex_weight
        )

        # get xr xi splitted parts
        xr = einsum_mul.real
        xi = einsum_mul.imag

        # apply silu
        real_act = self.act(xr)
        imag_act = self.act(xi)

        # activated complex
        activated_complex = torch.complex(real_act, imag_act)

        # apply ifft2d solver as notated
        iffted = torch.fft.ifft2(activated_complex, dim=(-2, -1))
        return iffted.real

# Random input tensor
x = torch.randn(1, 3, 64, 64)
# Instantiate EinFFT module
einfft = EinFFT(3, 64, 64)
# Apply EinFFT to get an output
output = einfft(x)
# Print output tensor
print(output)


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
