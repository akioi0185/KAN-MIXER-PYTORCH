import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from FourierKAN.fftKAN import NaiveFourierKANLayer

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class Transpose(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

    def forward(self, x):
        return x.transpose(self.c1, self.c2)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            NaiveFourierKANLayer(dim, hidden_dim),
            nn.Dropout(dropout),
            NaiveFourierKANLayer(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer_1(nn.Module):
    def __init__(self, cfg, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()
        image_h, image_w = pair(image_size)
        patch_h, patch_w = pair(patch_size)
        assert (image_h % patch_h) == 0 and (image_w % patch_w) == 0, 'image must be divisible by patch size'
        self.type = cfg.TASK.TYPE
        self.num_patch = (image_h // patch_h) * (image_w // patch_w)
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
             x = mixer_block(x)
        x = self.layer_norm(x)
        if self.type == 'classification':
            x = x.mean(dim=1)
            x = self.mlp_head(x)
        return x

class SegBlock(nn.Module):
    def __init__(self, in_channels, in_dim):
        super().__init__()
        self.in_channels = in_channels
        self.in_dim =  in_dim
        out_dim = 2*in_dim
        self.amplifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            NaiveFourierKANLayer(in_dim, out_dim)
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(out_dim),
            Rearrange('b n d -> b d n'),
            NaiveFourierKANLayer(in_channels, in_channels),
            Rearrange('b d n -> b n d')
        )

    def forward(self, x):
        x = self.amplifier(x)
        x = self.channel_mix(x)
        x = torch.reshape(x, (-1, 2*self.in_channels, self.in_dim))
        return x

class Segmentation( nn.Module ):
    def __init__(self, in_dim, in_channels, init_channels, depth):
        super().__init__()
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.init_channels = init_channels
        self.depth = depth
        
        self.Preprocess = nn.Sequential(
            Rearrange('b n d -> b d n'),
            NaiveFourierKANLayer(in_channels, init_channels),
            Rearrange('b d n -> b n d')
        )

        self.amplifier_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.amplifier_blocks.append(SegBlock(init_channels, in_dim))
            init_channels = 2*init_channels
        
    def forward(self, x):
        x = self.Preprocess(x)
        for amplifier_block in self.amplifier_blocks:
             x = amplifier_block(x)
        return x
        


# def MLPMixer(*, image_size, channels, patch_size, dim, depth,num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout=0.):
#     image_h, image_w = pair(image_size)
#     assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
#     num_patches = (image_h // patch_size) * (image_w // patch_size)
#     #chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

#     return nn.Sequential(
#         Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
#         nn.Linear((patch_size ** 2) * channels, dim),
#         *[nn.Sequential(
#             Transpose(1,2),
#             PreNormResidual(num_patches, FeedForward(num_patches, expansion_factor, dropout, NaiveFourierKANLayer)),
#             Transpose(1,2),
#             PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, NaiveFourierKANLayer))
#         ) for _ in range(depth)],
#         nn.LayerNorm(dim),
#         Reduce('b n c -> b c', 'mean'),
#         NaiveFourierKANLayer(dim, num_classes)
#     )