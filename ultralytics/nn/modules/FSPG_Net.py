import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Sequence
import warnings
from timm.models.layers import DropPath

warnings.filterwarnings('ignore')

try:
    from pytorch_wavelets import DWTForward
except ImportError:
    raise ImportError("Please install pytorch_wavelets: pip install pytorch-wavelets")

# =========================================================================
# 1. Global Ablation Configuration
# =========================================================================
GLOBAL_ABLATION_CFG = {
    'use_freq': True,  # Macro architecture: frequency domain switch
    'use_spatial': True,  # Macro architecture: spatial domain switch
    'spatial_mode': 'simple_global',  # Spatial ablation: 'full', 'simple_local' (remove LMP), 'simple_global' (remove GEP), 'simple_both'
    'pace_mode': 'full',  # Frequency module: 'none', 'amp_only', 'full'
    'spg_mode': 'full',  # Frequency module: 'none_keep', 'soft_thresh', 'struct_only', 'grad_only', 'full'
    'fusion_mode': 'full',  # Fusion mechanism: 'avg_weight', 'concat_add', 'no_sfcf', 'no_agsa', 'full'
    'wavelet_basis': 'haar'  # Hyperparameter: 'haar', 'db2', etc.
}

# =========================================================================
# 2. Basic Components & Utility Functions
# =========================================================================
def autopad(k, p=None, d=1):
    """
    Calculate automatic padding size for convolution
    Args:
        k: kernel size
        p: padding, if None compute automatically
        d: dilation rate
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """
    Make value divisible by divisor (widely used in MobileNet / ConvNeXt)
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


def build_norm_layer(cfg, num_features):
    """Build normalization layer by config dict"""
    if cfg is None:
        return 'None', nn.Identity()
    if cfg.get('type') == 'BN':
        return 'bn', nn.BatchNorm2d(num_features, momentum=cfg.get('momentum', 0.1), eps=cfg.get('eps', 1e-5))
    return 'norm', nn.BatchNorm2d(num_features)


class BasicConv2(nn.Module):
    """Basic Conv + BN + ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvModule(nn.Module):
    """Conv module with configurable norm and activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        bias = norm_cfg is None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1] if norm_cfg else nn.Identity()

        if act_cfg is None:
            self.act = nn.Identity()
        elif act_cfg.get('type') == 'SiLU':
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class LayerNorm(nn.Module):
    """LayerNorm for channels_first (B, C, H, W) format"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

# =========================================================================
# 3. Core Spatial Perception Modules (including ablation substitutes & original modules)
# =========================================================================
class SimpleLocalBlock(nn.Module):
    """
    For ablation study of LMP.
    Use ordinary residual convolution to replace LMP local branch
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = BasicConv2(dim, dim, 3, padding=1)
        self.conv2 = BasicConv2(dim, dim, 3, padding=1, relu=False)

    def forward(self, x):
        return F.relu(x + self.conv2(self.conv1(x)))


class StandardSelfAttention(nn.Module):
    """
    For ablation study of GEP.
    Use vanilla MultiheadAttention to replace GEP global branch
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Conv2d(dim, dim * 4, 1), nn.ReLU(True), nn.Conv2d(dim * 4, dim, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x_norm = self.norm1(x).flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        x = shortcut + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CAA(nn.Module):
    """Coordinate Attention Aggregation module"""
    def __init__(self, channels: int, h_kernel_size: int = 11, v_kernel_size: int = 11,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2),
                                 groups=channels, norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0),
                                 groups=channels, norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class InceptionBottleneck(nn.Module):
    """Inception-style multi-scale bottleneck with CAA attention"""
    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11), dilations: Sequence[int] = (1, 1, 1, 1, 1),
                 expansion: float = 1.0, add_identity: bool = True, with_caa: bool = True, caa_kernel_size: int = 11,
                 norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dw_convs = nn.ModuleList([
            ConvModule(hidden_channels, hidden_channels, k, 1, autopad(k, None, d), d,
                       groups=hidden_channels, norm_cfg=None, act_cfg=None)
            for k, d in zip(kernel_sizes, dilations)
        ])
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size, norm_cfg,
                              act_cfg) if with_caa else None
        self.add_identity = add_identity and in_channels == out_channels
        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)
        y = x
        x = sum(conv(x) for conv in self.dw_convs)
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
            x = x * y + x if self.add_identity else x * y
        return self.post_conv(x)


class LMP(nn.Module):
    """Local Multi-scale Perception block"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 3, 3, 3, 3),
            dilations: Sequence[int] = (1, 2, 3, 4, 5),
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            expansion: float = 1.0,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.norm1 = build_norm_layer(norm_cfg, in_channels)[1] if norm_cfg else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1] if norm_cfg else nn.Identity()

        self.block = InceptionBottleneck(
            in_channels, hidden_channels, kernel_sizes, dilations,
            expansion=1.0, add_identity=True,
            with_caa=with_caa, caa_kernel_size=caa_kernel_size,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )

        self.ffn = ConvModule(
            hidden_channels, out_channels, 1, 1, 0,
            norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)

        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.block(x)

        if self.layer_scale:
            x = self.gamma1.unsqueeze(-1).unsqueeze(-1) * x
        x = identity + self.drop_path(x) if self.add_identity else self.drop_path(x)

        identity = x
        x = self.norm2(x)
        x = self.ffn(x)

        if self.layer_scale:
            x = self.gamma2.unsqueeze(-1).unsqueeze(-1) * x
        x = identity + self.drop_path(x) if self.add_identity else self.drop_path(x)

        return x


class WATT(nn.Module):
    """Window Adaptive Relative Position Transformer"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        coords = torch.stack(torch.meshgrid([torch.arange(window_size)] * 2, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        rel_pos = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel_pos = rel_pos.permute(1, 2, 0).contiguous()
        rel_pos_enc = torch.sign(rel_pos) * torch.log(1. + rel_pos.abs())
        self.register_buffer("relative_positions", rel_pos_enc)
        self.meta = nn.Sequential(nn.Linear(2, 256), nn.ReLU(True), nn.Linear(256, num_heads))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        rel_bias = self.meta(self.relative_positions).permute(2, 0, 1).unsqueeze(0)
        attn = self.softmax(attn + rel_bias)
        return (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)


class GEP(nn.Module):
    """Global Enhanced Perception block based on shifted window attention"""
    def __init__(self, dim, num_heads=8, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.conv_v = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, padding_mode='reflect')
        self.conv_qk = nn.Conv2d(dim, dim * 2, 1)
        self.attn = WATT(dim, window_size, num_heads)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Conv2d(dim, dim * 4, 1), nn.ReLU(True), nn.Conv2d(dim * 4, dim, 1))

    def window_partition(self, x):
        """Split feature map into non-overlapping windows"""
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        return x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size ** 2, C)

    def window_reverse(self, windows, H, W):
        """Merge windows back to full feature map"""
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        return x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        V = self.conv_v(x)
        QK = self.conv_qk(x)
        QKV = torch.cat([QK, V], dim=1)
        B, C_qkv, H_ori, W_ori = QKV.shape
        pad_h = (self.window_size - H_ori % self.window_size) % self.window_size
        pad_w = (self.window_size - W_ori % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            QKV = F.pad(QKV, (0, pad_w, 0, pad_h), mode='constant', value=0)
        H_pad, W_pad = QKV.shape[2], QKV.shape[3]
        shifted = torch.roll(QKV, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        qkv_windows = self.window_partition(shifted)
        attn_windows = self.attn(qkv_windows)
        shifted_out = self.window_reverse(attn_windows, H_pad, W_pad)
        out = torch.roll(shifted_out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        out = out[:, :, :H_ori, :W_ori]
        x = self.proj(out) + shortcut
        x = x + self.mlp(self.norm2(x))
        return x


class SFCF(nn.Module):
    """Spatial-Frequency Cross Fusion attention module"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        kernels = [(1, 7), (1, 11), (1, 21), (7, 1), (11, 1), (21, 1)]
        pads = [(0, 3), (0, 5), (0, 10), (3, 0), (5, 0), (10, 0)]
        self.convs1 = nn.ModuleList([nn.Conv2d(dim, dim, k, padding=p, groups=dim) for k, p in zip(kernels, pads)])
        self.convs2 = nn.ModuleList([nn.Conv2d(dim, dim, k, padding=p, groups=dim) for k, p in zip(kernels, pads)])

    def forward(self, x1, x2):
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x1_norm, x2_norm = self.norm1(x1), self.norm2(x2)
        attn1 = sum(conv(x1_norm) for conv in self.convs1)
        attn2 = sum(conv(x2_norm) for conv in self.convs2)
        out1, out2 = self.proj(attn1), self.proj(attn2)
        B, C, H, W = out1.shape
        head_dim = C // self.num_heads
        q1 = out2.view(B, self.num_heads, head_dim, -1).transpose(2, 3)
        q2 = out1.view(B, self.num_heads, head_dim, -1).transpose(2, 3)
        k1, v1 = out1.view(B, self.num_heads, head_dim, -1).transpose(2, 3), out1.view(B, self.num_heads, head_dim,
                                                                                       -1).transpose(2, 3)
        k2, v2 = out2.view(B, self.num_heads, head_dim, -1).transpose(2, 3), out2.view(B, self.num_heads, head_dim,
                                                                                       -1).transpose(2, 3)
        q1, q2 = F.normalize(q1, dim=-1), F.normalize(q2, dim=-1)
        k1, k2 = F.normalize(k1, dim=-1), F.normalize(k2, dim=-1)
        o1 = (F.softmax(q1 @ k1.transpose(-2, -1), dim=-1) @ v1).transpose(2, 3).reshape(B, C, H, W)
        o2 = (F.softmax(q2 @ k2.transpose(-2, -1), dim=-1) @ v2).transpose(2, 3).reshape(B, C, H, W)
        return self.proj(o1) + self.proj(o2) + x1 + x2


class AGSA(nn.Module):
    """Adaptive Global-Spatial Attention fusion module"""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1x1_in = BasicConv2(2 * in_channels, in_channels, 1, relu=False)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.channel_att = nn.Sequential(BasicConv2(in_channels, in_channels, 1, relu=True), nn.Sigmoid())
        self.spatial_att_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.spatial_att_sigmoid = nn.Sigmoid()
        self.conv1x1_out = BasicConv2(in_channels, out_channels, 1, relu=True)

    def forward(self, feat_high, feat_low):
        feat_cat = torch.cat([feat_high, feat_low], dim=1)
        F_in = self.conv1x1_in(feat_cat)
        X_HAP_c = torch.mean(F_in, dim=2, keepdim=True)
        X_WAP_c = torch.mean(F_in, dim=3, keepdim=True)
        X_c = self.channel_att(X_HAP_c) * self.channel_att(X_WAP_c)
        X_GAP_s = torch.mean(F_in, dim=1, keepdim=True)
        X_GMP_s = torch.max(F_in, dim=1, keepdim=True)[0]
        X_s = self.spatial_att_sigmoid(self.spatial_att_conv(torch.cat([X_GAP_s, X_GMP_s], dim=1)))
        W_w = torch.sigmoid(X_c + X_s + F_in)
        F_fuse = self.alpha * feat_low * W_w + self.beta * feat_high * W_w + feat_low + feat_high
        return self.conv1x1_out(F_fuse)


class MSCM(nn.Module):
    """Multi-Scale Channel Modulation block"""
    def __init__(self, dim):
        super().__init__()
        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1), BasicConv2(dim, dim, 1), nn.Sigmoid())
        self.local_branch = nn.Sequential(BasicConv2(dim, dim, 3, padding=1, groups=dim), nn.Sigmoid())

    def forward(self, x):
        return self.global_branch(x) * self.local_branch(x)

# =========================================================================
# 4. Physics-inspired Frequency Denoising Modules (PACE & SPG)
# =========================================================================
class PACE_Ablation(nn.Module):
    """PACE amplitude-phase enhancement module with ablation options"""
    def __init__(self, channels):
        super().__init__()
        self.mode = GLOBAL_ABLATION_CFG['pace_mode']
        if self.mode != 'none':
            self.amp_conv = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.LeakyReLU(0.1, inplace=True),
                                          nn.Conv2d(channels, channels, 1))
        if self.mode == 'full':
            self.phase_to_amp_attn = nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1), nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True), nn.Conv2d(channels // 4, channels, 1), nn.Sigmoid()
            )

    def forward(self, x):
        if self.mode == 'none':
            return x
        B, C, H, W = x.shape
        fft_x = torch.fft.rfft2(x, norm='backward')
        amp, pha = torch.abs(fft_x), torch.angle(fft_x)
        amp_feature = self.amp_conv(amp)

        if self.mode == 'amp_only':
            amp_refined = amp + amp_feature
        else:
            phase_guidance = self.phase_to_amp_attn(pha)
            amp_refined = amp + amp_feature * phase_guidance

        fft_refined = torch.polar(amp_refined, pha)
        x_out = torch.fft.irfft2(fft_refined, s=(H, W), norm='backward')
        return x_out + x


class SPG_Ablation(nn.Module):
    """Structure Prior Gate module with ablation options"""
    def __init__(self, in_channels):
        super().__init__()
        self.mode = GLOBAL_ABLATION_CFG['spg_mode']
        if self.mode in ['soft_thresh', 'none_keep']:
            return

        self.structure_extractor = nn.Sequential(BasicConv2(in_channels, in_channels, 3, padding=1), nn.Sigmoid())
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.expand(in_channels, 1, 3, 3).clone())
        self.register_buffer('sobel_y', sobel_y.expand(in_channels, 1, 3, 3).clone())
        self.gate_generator = nn.Sequential(BasicConv2(in_channels * 2, in_channels, 1),
                                            nn.Conv2d(in_channels, in_channels, 1), nn.Sigmoid())

    def forward(self, x_low, x_high_bands):
        if self.mode == 'none_keep':
            return x_high_bands
        if self.mode == 'soft_thresh':
            thr = 0.1
            return torch.sign(x_high_bands) * F.relu(torch.abs(x_high_bands) - thr)

        S = self.structure_extractor(x_low)
        grad_x = F.conv2d(x_low, self.sobel_x, padding=1, groups=x_low.shape[1])
        grad_y = F.conv2d(x_low, self.sobel_y, padding=1, groups=x_low.shape[1])
        G = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        G = G / (G.max() + 1e-6)

        if self.mode == 'struct_only':
            G = torch.zeros_like(G)
        elif self.mode == 'grad_only':
            S = torch.zeros_like(S)

        priors = torch.cat([S, G], dim=1)
        gate = self.gate_generator(priors).unsqueeze(2)
        return x_high_bands * gate

# =========================================================================
# 5. Core Controller Module: FSF and FSFG
# =========================================================================
class FSF(nn.Module):
    """Frequency-Spatial Fusion Block"""
    def __init__(self, in_channels, out_channels, window_size=8, num_heads=8):
        super().__init__()
        self.proj_in = BasicConv2(in_channels, out_channels, 1)
        self.proj_res = BasicConv2(in_channels, out_channels, 1)

        self.use_freq = GLOBAL_ABLATION_CFG['use_freq']
        self.use_spatial = GLOBAL_ABLATION_CFG['use_spatial']
        self.spatial_mode = GLOBAL_ABLATION_CFG.get('spatial_mode', 'full')
        self.fusion_mode = GLOBAL_ABLATION_CFG['fusion_mode']

        if self.use_freq:
            self.wt = DWTForward(J=1, mode='zero', wave=GLOBAL_ABLATION_CFG['wavelet_basis'])
            self.pace_low = PACE_Ablation(out_channels)
            self.spg_gate = SPG_Ablation(out_channels)
            self.high_fuse = BasicConv2(out_channels * 3, out_channels, 1)
            self.mscm = MSCM(out_channels)

        if self.use_spatial:
            if self.spatial_mode in ['simple_local', 'simple_both']:
                self.local_extractor = SimpleLocalBlock(out_channels)
            else:
                self.local_extractor = LMP(out_channels, out_channels, drop_path_rate=0.1)

            if self.spatial_mode in ['simple_global', 'simple_both']:
                self.global_extractor = StandardSelfAttention(out_channels, num_heads)
            else:
                self.global_extractor = GEP(out_channels, num_heads, window_size)

        if self.fusion_mode == 'concat_add':
            self.simple_fuse_low = BasicConv2(out_channels * 2, out_channels, 1)
            self.simple_fuse_high = BasicConv2(out_channels * 2, out_channels, 1)
            self.simple_fuse_final = BasicConv2(out_channels * 2, out_channels, 1)
        elif self.fusion_mode in ['full', 'no_agsa', 'no_sfcf']:
            if self.fusion_mode in ['full', 'no_agsa']:
                self.sfcf_low, self.sfcf_high = SFCF(out_channels, num_heads), SFCF(out_channels, num_heads)
            if self.fusion_mode in ['full', 'no_sfcf']:
                self.agsa_fuse = AGSA(out_channels, out_channels)

    def forward(self, x):
        res = self.proj_res(x)
        x_proj = self.proj_in(x)

        # ---------------- Frequency stream ----------------
        if self.use_freq:
            yL, yH = self.wt(x_proj)
            yL_enhanced = self.pace_low(yL)
            yL_final = yL_enhanced * self.mscm(yL_enhanced + yL)

            yH_clean = self.spg_gate(yL_final, yH[0])
            B, C, _, H_sub, W_sub = yH_clean.shape
            F_high_clean = self.high_fuse(yH_clean.reshape(B, -1, H_sub, W_sub))

            F_low_up = F.interpolate(yL_final, size=x.shape[2:], mode='bilinear', align_corners=False)
            F_high_up = F.interpolate(F_high_clean, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            F_low_up, F_high_up = x_proj, x_proj

        # ---------------- Spatial stream ----------------
        if self.use_spatial:
            local_feat, global_feat = self.local_extractor(x_proj), self.global_extractor(x_proj)
        else:
            local_feat, global_feat = x_proj, x_proj

        # ---------------- Fusion stream ----------------
        if self.fusion_mode == 'avg_weight':
            feat_low = (F_low_up + global_feat) * 0.5
            feat_high = (F_high_up + local_feat) * 0.5
            fused = (feat_low + feat_high) * 0.5

        elif self.fusion_mode == 'concat_add':
            feat_low = self.simple_fuse_low(torch.cat([F_low_up, global_feat], dim=1))
            feat_high = self.simple_fuse_high(torch.cat([F_high_up, local_feat], dim=1))
            fused = self.simple_fuse_final(torch.cat([feat_low, feat_high], dim=1))

        elif self.fusion_mode == 'no_sfcf':
            feat_low, feat_high = F_low_up + global_feat, F_high_up + local_feat
            fused = self.agsa_fuse(feat_high, feat_low)

        elif self.fusion_mode == 'no_agsa':
            feat_low, feat_high = self.sfcf_low(F_low_up, global_feat), self.sfcf_high(F_high_up, local_feat)
            fused = feat_low + feat_high

        else:  # 'full'
            feat_low, feat_high = self.sfcf_low(F_low_up, global_feat), self.sfcf_high(F_high_up, local_feat)
            fused = self.agsa_fuse(feat_high, feat_low)

        return fused + res

# class AFM(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
#         super(AFM, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce
#         self.branch0 = nn.Sequential(
#             BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )
#
#         self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#
#         out = torch.cat((x0, x1, x2), 1)
#         out = self.ConvLinear(out)
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.relu(out)
#
#         return out

# class Concat2(nn.Module):
#     def __init__(self, dimension=1, Channel1=1, Channel2=1):
#         super(Concat2, self).__init__()
#         self.d = dimension
#         self.Channel1 = Channel1
#         self.Channel2 = Channel2
#         self.Channel_all = int(Channel1 + Channel2)
#         self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
#         self.epsilon = 0.0001
#
#     def forward(self, x):
#         N1, C1, H1, W1 = x[0].size()
#         N2, C2, H2, W2 = x[1].size()
#
#         w = self.w[:(C1 + C2)]
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)
#
#         x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
#         x2 = (weight[C1:] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
#         x = [x1, x2]
#         return torch.cat(x, self.d)
#
#
# class Concat3(nn.Module):
#     def __init__(self, dimension=1, Channel1=1, Channel2=1, Channel3=1):
#         super(Concat3, self).__init__()
#         self.d = dimension
#         self.Channel1 = Channel1
#         self.Channel2 = Channel2
#         self.Channel3 = Channel3
#         self.Channel_all = int(Channel1 + Channel2 + Channel3)
#         self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
#         self.epsilon = 0.0001
#
#     def forward(self, x):
#         N1, C1, H1, W1 = x[0].size()
#         N2, C2, H2, W2 = x[1].size()
#         N3, C3, H3, W3 = x[2].size()
#
#         w = self.w[:(C1 + C2 + C3)]
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)
#
#         x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
#         x2 = (weight[C1:(C1 + C2)] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
#         x3 = (weight[(C1 + C2):] * x[2].view(N3, H3, W3, C3)).view(N3, C3, H3, W3)
#         x = [x1, x2, x3]
#         return torch.cat(x, self.d)
#
#
# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
#         super(BasicConv, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
#                               stride=stride, padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True) if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
