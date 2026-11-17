
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_groups(channels, maximum=32):
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps):

        half = self.dimension // 2

        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half,
                device=timesteps.device,
                dtype=torch.float32
            )
            / max(half - 1, 1)
        )

        arguments = timesteps.float()[:, None] * frequencies[None, :]

        embedding = torch.cat([
            torch.sin(arguments),
            torch.cos(arguments)
        ], dim=-1)

        if self.dimension % 2:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            get_groups(in_channels), in_channels
        )

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(
            get_groups(out_channels), out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embedding):

        h = self.conv1(F.silu(self.norm1(x)))

        time = self.time_projection(time_embedding)
        h = h + time[:, :, None, None]

        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.skip(x)


class SelfAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.norm = nn.GroupNorm(
            get_groups(channels), channels
        )

        self.qkv = nn.Conv2d(
            channels, channels * 3, 1
        )

        self.projection = nn.Conv2d(
            channels, channels, 1
        )

    def forward(self, x):

        batch, channels, height, width = x.shape

        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)

        q = q.reshape(batch, channels, height * width).transpose(1, 2)
        k = k.reshape(batch, channels, height * width)
        v = v.reshape(batch, channels, height * width).transpose(1, 2)

        attention = torch.bmm(q, k) * (channels ** -0.5)
        attention = attention.softmax(dim=-1)

        h = torch.bmm(attention, v)

        h = h.transpose(1, 2).reshape(
            batch, channels, height, width
        )

        return x + self.projection(h)


class UNet(nn.Module):
    """
    Time-conditioned residual U-Net for DDPM.

    Supports arbitrary image channels and spatial sizes
    divisible by four.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        time_dim=256,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.input_conv = nn.Conv2d(
            in_channels, c1, 3, padding=1
        )

        # Encoder
        self.enc1 = ResidualBlock(c1, c1, time_dim)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)

        self.enc2 = ResidualBlock(c2, c2, time_dim)
        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)

        # Bottleneck
        self.mid1 = ResidualBlock(c3, c3, time_dim)
        self.attention = SelfAttention(c3)
        self.mid2 = ResidualBlock(c3, c3, time_dim)

        # Decoder
        self.up1 = nn.Conv2d(c3, c2, 3, padding=1)
        self.dec1 = ResidualBlock(c2 * 2, c2, time_dim)

        self.up2 = nn.Conv2d(c2, c1, 3, padding=1)
        self.dec2 = ResidualBlock(c1 * 2, c1, time_dim)

        self.output_norm = nn.GroupNorm(
            get_groups(c1), c1
        )

        self.output_conv = nn.Conv2d(
            c1, out_channels, 3, padding=1
        )

    def forward(self, x, timesteps):

        if x.ndim != 4:
            raise ValueError("Expected [B, C, H, W] input.")

        if x.shape[1] != self.in_channels:
            raise ValueError("Input channel mismatch.")

        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("Image dimensions must be divisible by 4.")

        t = self.time_embedding(timesteps)

        x0 = self.input_conv(x)

        e1 = self.enc1(x0, t)

        d1 = self.down1(e1)
        e2 = self.enc2(d1, t)

        d2 = self.down2(e2)

        h = self.mid1(d2, t)
        h = self.attention(h)
        h = self.mid2(h, t)

        h = F.interpolate(
            h, size=e2.shape[-2:], mode="nearest"
        )
        h = self.up1(h)

        h = self.dec1(
            torch.cat([h, e2], dim=1), t
        )

        h = F.interpolate(
            h, size=e1.shape[-2:], mode="nearest"
        )
        h = self.up2(h)

        h = self.dec2(
            torch.cat([h, e1], dim=1), t
        )

        return self.output_conv(F.silu(self.output_norm(h)))
