import torch
from torch import nn, Tensor
from zeta.nn import MambaBlock
from simba_torch.ein_fft import EinFFTText
from einops.layers.torch import Rearrange, Reduce


def posemb_sincos_2d(
    h, w, dim, temperature: int = 10000, dtype=torch.float32
):
    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )
    assert (
        dim % 4
    ) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class SimbaBlock(nn.Module):
    """
    SimbaBlock is a module that represents a block in the Simba model.

    Args:
        dim (int): The input dimension.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        d_state (int, optional): The state dimension. Defaults to 64.
        d_conv (int, optional): The convolution dimension. Defaults to 64.
        num_classes (int, optional): The number of classes. Defaults to 64.

    Attributes:
        dim (int): The input dimension.
        d_state (int): The state dimension.
        d_conv (int): The convolution dimension.
        num_classes (int): The number of classes.
        dropout (nn.Dropout): The dropout layer.
        mamba (MambaBlock): The MambaBlock module.
        einfft (EinFFTText): The EinFFTText module.

    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        d_state: int = 64,
        d_conv: int = 64,
        num_classes: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)

        # Mamba Block
        self.mamba = MambaBlock(
            dim=self.dim,
            depth=1,
            d_state=self.d_state,
            d_conv=self.d_conv,
        )

        # EinFFT
        self.einfft = EinFFTText(
            sequence_length=num_classes,
            dim=self.dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SimbaBlock module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        # patch
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
        out = out + residual_new
        return out.real


# # Forward pass with images
# img = torch.randn(1, 64, 3)

# # Create model
# model = Simba(
#     dim = 3,
#     dropout = 0.1,
#     heads=8,
#     dim_head=64,
#     d_state=64,
#     d_conv=64,
# )

# # out
# out = model(img)

# # Print output shape
# print(out.shape)


class Simba(nn.Module):
    """
    Simba model implementation.

    Args:
        dim (int): Dimension of the model.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        d_state (int, optional): Dimension of the state. Defaults to 64.
        d_conv (int, optional): Dimension of the convolutional layer. Defaults to 64.
        num_classes (int, optional): Number of output classes. Defaults to 64.
        depth (int, optional): Number of Simba blocks. Defaults to 8.
        patch_size (int, optional): Size of the patches. Defaults to 16.
        image_size (int, optional): Size of the input image. Defaults to 224.
        channels (int, optional): Number of input channels. Defaults to 3.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        d_state: int = 64,
        d_conv: int = 64,
        num_classes: int = 64,
        depth: int = 8,
        patch_size: int = 16,
        image_size: int = 224,
        channels: int = 3,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_classes = num_classes
        self.depth = depth
        self.patch_size = patch_size
        self.image_size = image_size
        self.use_pos_emb = use_pos_emb

        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        patch_dim = channels * patch_height * patch_width

        # Layers
        self.simba_blocks = nn.ModuleList(
            [
                SimbaBlock(
                    dim=self.dim,
                    dropout=self.dropout,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    num_classes=self.num_classes,
                )
                for _ in range(self.depth)
            ]
        )

        # Patch Embedding
        self.to_patch = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.dim),
            nn.LayerNorm(dim),
        )

        # To latent
        self.to_latent = nn.Identity()

        # Head
        self.output_head = nn.Sequential(
            Reduce("b n d -> b d", "mean"),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes),
        )

        # Positional embeddings
        self.pos_emb = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Simba model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # device
        device = x.device

        # Patch
        x = self.to_patch(x)
        print(f"Patch: {x.shape}")

        # Maybe add positional embeddings here
        if self.use_pos_emb:
            x += self.pos_emb.to(device, dtype=x.dtype)

        # Loop through simba blocks
        for layer in self.simba_blocks:
            x = layer(x)
            print(f"Layer: {x.shape}")

        # To latent
        x = self.to_latent(x)
        print(x.shape)

        return self.output_head(x)
