import torch
from torch import nn, Tensor


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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        # silu
        self.act = nn.SiLU()

        # complex weights for channel-wise transformation
        self.complex_weight = nn.Parameter(
            torch.randn(
                in_channels, out_channels, dtype=torch.complex64
            )
        )

        # Real weight
        self.real_weight = nn.Parameter(
            torch.randn(in_channels, out_channels)
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

        # get xr xi splitted parts
        xr = fast_fouried.real
        xi = fast_fouried.imag
        print(xr.shape, xi.shape)

        # complex-valued multiplication
        einsum_mul = torch.einsum(
            "bchw,cf->bchw", xr, self.complex_weight
        ) + torch.einsum("bchw,cf->bchw", xi, self.complex_weight)
        print(einsum_mul.shape)

        xr = einsum_mul.real
        xi = einsum_mul.imag

        # apply silu
        real_act = self.act(xr)
        imag_act = self.act(xi)

        # EMM with the weights use torch split instead
        emmed = torch.einsum(
            "bchw,cf->bchw", real_act, self.real_weight
        ) + torch.einsum(
            "bchw,cf->bchw", imag_act, self.complex_weight
        )
        print(emmed.shape)
        print(emmed.dtype)

        # activated complex
        # activated_complex = torch.complex(emmed, emmed)

        # apply ifft2d solver as notated
        iffted = torch.fft.ifft2(emmed + emmed, dim=(-2, -1))
        return iffted.real


# # Random input tensor
# x = torch.randn(1, 3, 64, 64)
# # Instantiate EinFFT module
# einfft = EinFFT(3, 64, 64)
# # Apply EinFFT to get an output
# output = einfft(x)
# # Print output tensor
# print(output)


class EinFFTText(nn.Module):
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
        sequence_length: int,
        dim: int,
    ):
        super().__init__()
        self.dim = dim

        # silu
        self.act = nn.SiLU()

        # complex weights for channel-wise transformation
        self.complex_weight = nn.Parameter(
            torch.randn(sequence_length, dim, dtype=torch.complex64)
        )

        # Real weight
        self.real_weight = nn.Parameter(
            torch.randn(sequence_length, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the EinFFT module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, dimension).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, dimension).

        """
        b, s, d = x.shape

        # apply 1D FFTSolver, transform input tensor to frequency domain
        fast_fouried = torch.fft.fft(x, dim=-2)

        # get xr xi splitted parts
        xr = fast_fouried.real
        xi = fast_fouried.imag

        # complex-valued multiplication
        einsum_mul = torch.einsum(
            "bsd,cf->bsd", xr, self.complex_weight
        ) + torch.einsum("bsd,cf->bsd", xi, self.complex_weight)

        xr = einsum_mul.real
        xi = einsum_mul.imag

        # apply silu
        real_act = self.act(xr)
        imag_act = self.act(xi)

        # EMM with the weights use torch split instead
        emmed = torch.einsum(
            "bsd,cf->bsd", real_act, self.real_weight
        ) + torch.einsum("bsd,cf->bsd", imag_act, self.complex_weight)

        # apply ifft solver as notated
        iffted = torch.fft.ifft(emmed + emmed, dim=-2)
        return iffted.real


# Random input tensor
# x = torch.randn(1, 3, 64)

# # Instantiate EinFFT module
# einfft = EinFFTText(3, 64)

# # Apply EinFFT to get an output
# output = einfft(x)
# print(output.shape)
