#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import torch
import torch.nn as nn
from segmentation_models_pytorch import decoders
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder

from einops import rearrange, reduce
import torch.nn.functional as F
from hamburger.ham_spetral import get_hams


try:
    import ptwt
    PTWT_AVAILABLE = True
except ImportError:
    ptwt = None
    PTWT_AVAILABLE = False


class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2', level=1):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        lp = torch.tensor([
            0.4829629131445341,
            0.8365163037378079,
            0.2241438680420134,
            -0.12940952255126034
        ])
        self.register_buffer('db2_lp', lp)

    def forward(self, x):
        # x: (BHW, C, S)
        bhw, c, s = x.shape
        assert self.level == 1, "This implementation assumes level=1 (S/2 tokens)"

        if s % 2 != 0:
            x = F.pad(x, (0, 1))
        out_len = x.shape[-1] // 2

        if not PTWT_AVAILABLE:
            kernel = self.db2_lp.view(1, 1, 4).repeat(c, 1, 1)
            x = F.conv1d(x, kernel, stride=2, padding=1, groups=c)
            return x

        lf_tensor = x.new_zeros((bhw, c, out_len))
        for i in range(c):
            coeffs = ptwt.wavedec(x[:, i, :], wavelet=self.wavelet, level=self.level)
            lf = coeffs[0]
            if lf.shape[-1] > out_len:
                lf = lf[..., :out_len]
            elif lf.shape[-1] < out_len:
                lf = F.pad(lf, (0, out_len - lf.shape[-1]))
            lf_tensor[:, i, :] = lf
        return lf_tensor


class SKFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        # x: (b g c h w)
        b, g, c, h, w = x.shape
        pooled = x.mean(dim=(-1, -2))  # b g c
        weights = self.fc2(F.relu(self.fc1(pooled)))  # b g 1
        weights = weights.softmax(dim=1)
        weights = weights.view(b, g, 1, 1, 1)
        return (x * weights).sum(dim=1)


class Hamburger(nn.Module):
    def __init__(self, in_c, args=None):
        super().__init__()
        ham_type = getattr(args, 'HAM_TYPE', 'NMF')

        C = in_c
        # self.norm = nn.BatchNorm1d(C)
        num_groups = self._pick_group_count(C)
        # GroupNorm is more stable than BN for small batch sizes.
        self.norm = nn.GroupNorm(num_groups, C)
        if ham_type == 'NMF':
            self.lower_bread = nn.Sequential(nn.Conv1d(C, C, 1),
                                             nn.ReLU(inplace=True))
        else:
            self.lower_bread = nn.Conv1d(C, C, 1)

        HAM = get_hams(ham_type)
        self.ham = HAM(args)
        self.ham.D = in_c

        self.upper_bread = nn.Conv1d(C, C, 1, bias=False)
        self.shortcut = nn.Sequential()

    @staticmethod
    def _pick_group_count(channels):
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x):  # x: (b h w) c s
        ham_x = self.norm(x)
        ham_x = self.lower_bread(ham_x)
        ham_x = self.ham(ham_x)

        out = F.relu(x + ham_x, inplace=True)
        return out

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)


class SMD(nn.Module):
    def __init__(self, in_channels, hidden_feature, spe_reduction=1,
                 dim_reduction=4, kernel_size=1, lf_alpha_init=1.0,
                 use_multiscale=False, use_hf=False, ham_gate_init=1.0,
                 ham_group_size=0, sparse_reg_weight=0.0, orth_reg_weight=0.0,
                 tv_reg_weight=0.0, ham_lf_beta=0.1):
        super(SMD, self).__init__()
        self.spe_reduction = spe_reduction
        self.hidden_feature = hidden_feature
        self.use_multiscale = use_multiscale
        self.use_hf = use_hf
        self.ham_group_size = ham_group_size
        self.sparse_reg_weight = sparse_reg_weight
        self.orth_reg_weight = orth_reg_weight
        self.tv_reg_weight = tv_reg_weight
        self.ham_lf_beta_raw = nn.Parameter(torch.tensor(-2.0))

        self.depthwiseconv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_feature,
                                       kernel_size=spe_reduction, stride=spe_reduction)

        num_groups = self._pick_group_count(hidden_feature)
        self.spec_local = nn.Sequential(
            nn.Conv1d(hidden_feature, hidden_feature, kernel_size=3, padding=1, groups=hidden_feature),
            nn.GroupNorm(num_groups, hidden_feature),
            nn.GELU(),
        )
        if self.use_multiscale:
            # Multi-scale depthwise convs to enrich local spectral patterns.
            self.spec_local_k5 = nn.Conv1d(hidden_feature, hidden_feature, kernel_size=5, padding=2, groups=hidden_feature)
            self.spec_local_k7 = nn.Conv1d(hidden_feature, hidden_feature, kernel_size=7, padding=3, groups=hidden_feature)

        self.qkv = nn.Linear(hidden_feature, hidden_feature * 3)
        self.ham = Hamburger(hidden_feature)
        # Learnable scaling on HAM output to prevent over-dominance.
        self.ham_gate = nn.Parameter(torch.tensor(float(ham_gate_init)))
        if self.ham_group_size and self.ham_group_size > 0:
            # 1x1 fusion to reduce boundary artifacts between HAM groups.
            self.ham_fusion = nn.Conv1d(hidden_feature, hidden_feature, kernel_size=1)
        self.wavelet_transform = WaveletTransform(wavelet='db2', level=1)

        self.lf_gate = nn.Sequential(
            nn.Linear(hidden_feature * 3, hidden_feature // 2),
            nn.GELU(),
            nn.Linear(hidden_feature // 2, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _pick_group_count(channels):
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x):
        b, c, s, h, w = x.shape
        x = rearrange(x, 'b c s h w -> (b s) c h w')
        x = self.depthwiseconv(x)
        h_, w_ = x.shape[-2], x.shape[-1]

        x_base = rearrange(x, '(b s) c h w -> (b h w) c s', s=s)

        x_local = self.spec_local(x_base)
        if self.use_multiscale:
            # Fuse multi-scale spectral details with a light residual sum.
            x_local = x_local + self.spec_local_k5(x_base) + self.spec_local_k7(x_base)

        qkv = self.qkv(x_local.transpose(1, 2))
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        context = torch.matmul(k.transpose(-2, -1), v)
        x_attn = torch.matmul(q, context).transpose(1, 2)

        x_lf = self.wavelet_transform(x_base)
        x_lf = F.interpolate(x_lf, size=s, mode='linear', align_corners=False)
        if self.use_hf:
            # High-frequency branch from residual between original and low-frequency.
            x_hf = x_base - x_lf
            x_lf = x_lf + x_hf

        stat_feat = torch.cat(
            [x_base.mean(-1), x_base.std(-1, unbiased=False), x_lf.mean(-1)],
            dim=-1
        )
        alpha = self.lf_gate(stat_feat).unsqueeze(-1)

        # Constrain beta to (0, 0.2) to keep LF guidance subtle and stable.
        beta = torch.sigmoid(self.ham_lf_beta_raw) * 0.2
        ham_in = x_local + x_attn + (beta * alpha * x_lf)
        if self.ham_group_size and self.ham_group_size > 0:
            # Split spectral bands into groups for localized HAM modeling.
            chunks = torch.split(ham_in, self.ham_group_size, dim=-1)
            ham_chunks = [self.ham(chunk) for chunk in chunks]
            x_ham = torch.cat(ham_chunks, dim=-1)
            x_ham = self.ham_fusion(x_ham)
        else:
            x_ham = self.ham(ham_in)
        x_ham = self.ham_gate * x_ham
        # Cache HAM output for spectral smoothness regularization.
        self._last_ham_output = x_ham

        out = x_base + x_ham + alpha * x_lf
        out = rearrange(out, '(b h w) c s -> b c s h w', b=b, h=h_, w=w_)
        return out

    def regularization_loss(self):
        # Optional extra loss for sparse/orthogonal priors (call in training loop).
        param = next(self.parameters())
        loss = param.new_tensor(0.0)
        ham_impl = getattr(self.ham, "ham", None)
        if self.sparse_reg_weight > 0.0 and hasattr(ham_impl, "codes"):
            # Encourage sparse codes when HAM exposes them.
            loss = loss + self.sparse_reg_weight * ham_impl.codes.abs().mean()
        if self.orth_reg_weight > 0.0 and hasattr(ham_impl, "bases"):
            # Encourage orthogonal bases to reduce redundancy.
            bases = ham_impl.bases
            gram = torch.matmul(bases.transpose(-2, -1), bases)
            eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
            loss = loss + self.orth_reg_weight * (gram - eye).pow(2).mean()
        if self.tv_reg_weight > 0.0 and hasattr(self, "_last_ham_output"):
            # Spectral total variation on HAM output to enforce smoothness.
            ham_x = self._last_ham_output
            tv = (ham_x[:, :, 1:] - ham_x[:, :, :-1]).abs().mean()
            loss = loss + self.tv_reg_weight * tv
        return loss

# --- Previous SMD implementation retained for reference (commented as requested) ---
# class SMD(nn.Module):
#     def __init__(self, in_channels, hidden_feature, spe_reduction=1,
#                  dim_reduction=4, kernel_size=1, lf_alpha_init=1.0):
#         super(SMD, self).__init__()
#         self.dim_reduction = dim_reduction
#         self.spe_reduction = spe_reduction
#
#         self.depthwiseconv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_feature, kernel_size=spe_reduction,
#                                        stride=spe_reduction, groups=in_channels)
#         self.ham = Hamburger(hidden_feature)
#         self.wavelet_transform = WaveletTransform(wavelet='db2', level=1)
#         self.lf_alpha = nn.Parameter(torch.tensor(float(lf_alpha_init)))
#         self.spectral_ffn = nn.Sequential(
#                                           nn.BatchNorm1d(hidden_feature),
#                                           nn.Conv1d(hidden_feature, hidden_feature, kernel_size=kernel_size, stride=1,
#                                                     padding=kernel_size // 2 if kernel_size != 1 else 0),
#                                           nn.GELU(),
#                                           nn.Conv1d(hidden_feature, hidden_feature, kernel_size=kernel_size, stride=1,
#                                                     padding=kernel_size // 2 if kernel_size != 1 else 0),
#                                           )
#
#     def forward(self, x):
#         b, c, s, h, w = x.shape
#         x = rearrange(x, 'b c s h w -> (b s) c h w')
#         x = self.depthwiseconv(x)
#         h_, w_ = x.shape[-2], x.shape[-1]
#
#         x = rearrange(x, '(b s) c h w -> (b h w) c s', s=s)
#         x_lf = self.wavelet_transform(x)
#         x_lf = F.interpolate(x_lf, size=x.shape[-1], mode='linear', align_corners=False)
#         alpha = torch.sigmoid(self.lf_alpha)  # sigmoid 残差
#         x = x + alpha * x_lf
#         x = self.ham(x)
#         o = self.spectral_ffn(x) + x
#         o = rearrange(o, '(b h w) c s -> b c s h w', b=b, h=h_, w=w_)
#         return o


# NOTE: ABLATION switches (mirrors basemodel_double_ablation2.py)
class SMD_Ablation(nn.Module):
    """
    ABLATION SWITCHES (default keeps behavior close to SMD):
      - enable_wavelet: use LF wavelet branch
      - wavelet_source: 'base' | 'local' | 'attn' (where LF is computed from)
      - enable_attn: use qkv attention branch
      - enable_local: use local spectral conv branch
      - enable_ham: use HAM block
      - enable_multiscale: use k5/k7 local convs
      - enable_hf: add HF residual (x_base - x_lf)
      - enable_lf_gate: use alpha gate (lf_gate); otherwise alpha=1
      - enable_ham_gate: use ham_gate scaling; otherwise scale=1
      - wavelet_in_ham: inject LF into HAM input (False matches SMD3-style)
      - wavelet_basis: 'db2' | 'haar'
      - wavelet_mode: 'lf' | 'hf' | 'lf_hf_concat'
      - output_mode: 'base_ham_lf' | 'no_base'
    """
    def __init__(self, in_channels, hidden_feature, spe_reduction=1,
                 dim_reduction=4, kernel_size=1, lf_alpha_init=1.0,
                 use_multiscale=False, use_hf=False, ham_gate_init=1.0,
                 ham_group_size=0, sparse_reg_weight=0.0, orth_reg_weight=0.0,
                 tv_reg_weight=0.0, ham_lf_beta=0.1,
                 # --- ablation flags ---
                 enable_wavelet=True,
                 wavelet_source='base',   # 'base' | 'local' | 'attn'
                 enable_attn=True,
                 enable_local=True,
                 enable_ham=True,
                 enable_multiscale=True,
                 enable_hf=False,
                 enable_lf_gate=True,
                 enable_ham_gate=True,
                 wavelet_in_ham=True,
                 wavelet_basis='db2',
                 wavelet_mode='lf',
                 output_mode='base_ham_lf'):
        super(SMD_Ablation, self).__init__()
        self.spe_reduction = spe_reduction
        self.hidden_feature = hidden_feature
        self.use_multiscale = use_multiscale
        self.use_hf = use_hf
        self.ham_group_size = ham_group_size
        self.sparse_reg_weight = sparse_reg_weight
        self.orth_reg_weight = orth_reg_weight
        self.tv_reg_weight = tv_reg_weight
        self.ham_lf_beta_raw = nn.Parameter(torch.tensor(-2.0))

        # ABLATION switches
        self.enable_wavelet = enable_wavelet
        self.wavelet_source = wavelet_source
        self.enable_attn = enable_attn
        self.enable_local = enable_local
        self.enable_ham = enable_ham
        self.enable_multiscale = enable_multiscale
        self.enable_hf = enable_hf
        self.enable_lf_gate = enable_lf_gate
        self.enable_ham_gate = enable_ham_gate
        self.wavelet_in_ham = wavelet_in_ham
        self.wavelet_basis = wavelet_basis
        self.wavelet_mode = wavelet_mode
        self.output_mode = output_mode

        self.depthwiseconv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_feature,
                                       kernel_size=spe_reduction, stride=spe_reduction)

        num_groups = self._pick_group_count(hidden_feature)
        self.spec_local = nn.Sequential(
            nn.Conv1d(hidden_feature, hidden_feature, kernel_size=3, padding=1, groups=hidden_feature),
            nn.GroupNorm(num_groups, hidden_feature),
            nn.GELU(),
        )
        if self.enable_multiscale or self.use_multiscale:
            # Multi-scale depthwise convs to enrich local spectral patterns.
            self.spec_local_k5 = nn.Conv1d(hidden_feature, hidden_feature, kernel_size=5, padding=2, groups=hidden_feature)
            self.spec_local_k7 = nn.Conv1d(hidden_feature, hidden_feature, kernel_size=7, padding=3, groups=hidden_feature)

        self.qkv = nn.Linear(hidden_feature, hidden_feature * 3)
        self.ham = Hamburger(hidden_feature)
        # Learnable scaling on HAM output to prevent over-dominance.
        self.ham_gate = nn.Parameter(torch.tensor(float(ham_gate_init)))
        if self.ham_group_size and self.ham_group_size > 0:
            # 1x1 fusion to reduce boundary artifacts between HAM groups.
            self.ham_fusion = nn.Conv1d(hidden_feature, hidden_feature, kernel_size=1)
        self.wavelet_transform = WaveletTransform(wavelet=wavelet_basis, level=1)
        self.lf_hf_fuse = None
        if wavelet_mode == 'lf_hf_concat':
            self.lf_hf_fuse = nn.Conv1d(hidden_feature * 2, hidden_feature, kernel_size=1)

        self.lf_gate = nn.Sequential(
            nn.Linear(hidden_feature * 3, hidden_feature // 2),
            nn.GELU(),
            nn.Linear(hidden_feature // 2, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _pick_group_count(channels):
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x):
        b, c, s, h, w = x.shape
        x = rearrange(x, 'b c s h w -> (b s) c h w')
        x = self.depthwiseconv(x)
        h_, w_ = x.shape[-2], x.shape[-1]

        x_base = rearrange(x, '(b s) c h w -> (b h w) c s', s=s)

        # --- local branch ---
        if self.enable_local:
            x_local = self.spec_local(x_base)
            if self.enable_multiscale or self.use_multiscale:
                x_local = x_local + self.spec_local_k5(x_base) + self.spec_local_k7(x_base)
        else:
            x_local = x_base

        # --- attention branch ---
        if self.enable_attn:
            qkv = self.qkv(x_local.transpose(1, 2))
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)
            context = torch.matmul(k.transpose(-2, -1), v)
            x_attn = torch.matmul(q, context).transpose(1, 2)
        else:
            x_attn = 0

        # --- wavelet LF branch ---
        if self.enable_wavelet:
            if self.wavelet_source == 'local':
                x_wave = x_local
            elif self.wavelet_source == 'attn':
                x_wave = x_attn if self.enable_attn else x_local
            else:
                x_wave = x_base
            x_lf = self.wavelet_transform(x_wave)
            x_lf = F.interpolate(x_lf, size=s, mode='linear', align_corners=False)
            x_hf = None
            if self.enable_hf or self.wavelet_mode in ('hf', 'lf_hf_concat'):
                x_hf = x_base - x_lf
            if self.wavelet_mode == 'hf':
                x_lf = x_hf
            elif self.enable_hf and self.wavelet_mode == 'lf':
                # Backward-compat: original behavior added HF residual back to LF (no-op overall).
                x_lf = x_lf + x_hf
            elif self.wavelet_mode == 'lf_hf_concat':
                if self.lf_hf_fuse is None:
                    raise RuntimeError("lf_hf_fuse is not initialized. Set wavelet_mode='lf_hf_concat' at init.")
                x_lf = self.lf_hf_fuse(torch.cat([x_lf, x_hf], dim=1))
        else:
            x_lf = 0

        # --- LF gating ---
        if self.enable_lf_gate and self.enable_wavelet:
            stat_feat = torch.cat(
                [x_base.mean(-1), x_base.std(-1, unbiased=False), x_lf.mean(-1)],
                dim=-1
            )
            alpha = self.lf_gate(stat_feat).unsqueeze(-1)
        else:
            alpha = 1.0

        # Constrain beta to (0, 0.2) to keep LF guidance subtle and stable.
        beta = torch.sigmoid(self.ham_lf_beta_raw) * 0.2
        if self.enable_wavelet and self.wavelet_in_ham:
            ham_in = x_local + x_attn + (beta * alpha * x_lf)
        else:
            ham_in = x_local + x_attn

        # --- HAM branch ---
        if self.enable_ham:
            if self.ham_group_size and self.ham_group_size > 0:
                chunks = torch.split(ham_in, self.ham_group_size, dim=-1)
                ham_chunks = [self.ham(chunk) for chunk in chunks]
                x_ham = torch.cat(ham_chunks, dim=-1)
                x_ham = self.ham_fusion(x_ham)
            else:
                x_ham = self.ham(ham_in)
        else:
            x_ham = 0

        if self.enable_ham_gate:
            x_ham = self.ham_gate * x_ham

        # Cache HAM output for spectral smoothness regularization.
        if self.enable_ham:
            self._last_ham_output = x_ham

        if self.output_mode == 'no_base':
            out = x_ham + (alpha * x_lf if self.enable_wavelet else 0)
        else:
            out = x_base + x_ham + (alpha * x_lf if self.enable_wavelet else 0)
        out = rearrange(out, '(b h w) c s -> b c s h w', b=b, h=h_, w=w_)
        return out

    def regularization_loss(self):
        if not self.enable_ham:
            return next(self.parameters()).new_tensor(0.0)
        # Optional extra loss for sparse/orthogonal priors (call in training loop).
        param = next(self.parameters())
        loss = param.new_tensor(0.0)
        ham_impl = getattr(self.ham, "ham", None)
        if self.sparse_reg_weight > 0.0 and hasattr(ham_impl, "codes"):
            loss = loss + self.sparse_reg_weight * ham_impl.codes.abs().mean()
        if self.orth_reg_weight > 0.0 and hasattr(ham_impl, "bases"):
            bases = ham_impl.bases
            gram = torch.matmul(bases.transpose(-2, -1), bases)
            eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
            loss = loss + self.orth_reg_weight * (gram - eye).pow(2).mean()
        if self.tv_reg_weight > 0.0 and hasattr(self, "_last_ham_output"):
            ham_x = self._last_ham_output
            tv = (ham_x[:, :, 1:] - ham_x[:, :, :-1]).abs().mean()
            loss = loss + self.tv_reg_weight * tv
        return loss


class merge_block(nn.Module):
    def __init__(self, spa_in_channels, spe_in_channels, bands_group=4,
                 merge_spe_downsample=2):
        super(merge_block, self).__init__()
        self.bands_group = bands_group
        self.merge_spe_downsample = merge_spe_downsample

        self.merge_conv_spa = nn.Sequential(nn.Conv2d(spa_in_channels + spe_in_channels,
                                                      spa_in_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, spa_input, spe_input):  # (b g) c h w , b c s h w
        spa_input = rearrange(spa_input, '(b g) c h w -> b g c h w', g=self.bands_group)
        spe2spa_input = reduce(spe_input, 'b c (s1 p) h w -> b s1 c h w', 'mean',
                               p=self.merge_spe_downsample)
        spa_input = self.merge_conv_spa(
            rearrange(torch.cat((spa_input, spe2spa_input), 2), 'b g c h w -> (b g) c h w'))

        return spa_input, spe_input



class SpatialSpetralMixStream(nn.Module):
    def __init__(self, spectral_channels, linkpos=[0, 0, 1, 0, 1], bands_group=15,
                 backbone='resnet34', spectral_hidden_feature=32,
                 spatial_pretrain=False, spe_kernel_size=1,
                 merge_spe_downsample=[2, 1],
                 spa_reduction=[4, 4], decode_choice='unet',
                 tv_reg_weight=0.0, ham_lf_beta=0.1,
                 hw=[64, 64], rank=4,
                 smd_ablation_kwargs=None, sk_reduction=16):  # NOTE: LD removed; ablation switches added
        # NOTE: SK fusion (option 1) enabled

        super(SpatialSpetralMixStream, self).__init__()
        self.spectral_channels = spectral_channels
        self.linkpos = linkpos
        self.spe_feature_dim = [1, spectral_hidden_feature * 1, spectral_hidden_feature * 2,
                                spectral_hidden_feature * 4,
                                spectral_hidden_feature * 8, spectral_hidden_feature * 16]

        self.bands_group = bands_group
        # NOTE: LD removed; no attention_group used
        smd_ablation_kwargs = smd_ablation_kwargs or {}  # NOTE: ablation switches

        spectrallayer_num = np.array(linkpos).nonzero()[0]
        self.spatial_backbone = get_encoder(name=backbone,
                                            in_channels=spectral_channels // bands_group,
                                            depth=5, weights='imagenet' if spatial_pretrain else None,
                                            output_stride=16 if decode_choice == 'deeplabv3plus' else 32)
        self.spa_feature_dim = list(self.spatial_backbone.out_channels[1:])
        self.encoder_stages = self.spatial_backbone.get_stages()

        self.sk_fusion = SKFusion(self.spa_feature_dim[-1], reduction=sk_reduction)  # NOTE: SK fusion

        self.merge_stages = nn.ModuleList([merge_block(self.spa_feature_dim[i - 1], self.spe_feature_dim[idx + 1],
                                                       bands_group=bands_group,
                                                       merge_spe_downsample=merge_spe_downsample[idx])
                                           for idx, i in enumerate(spectrallayer_num)])

        self.spe_encoder_stages = nn.ModuleList([SMD_Ablation(
            self.spe_feature_dim[idx], self.spe_feature_dim[idx + 1],
            spa_reduction[idx],
            kernel_size=spe_kernel_size,
            use_multiscale=True,
            tv_reg_weight=tv_reg_weight,
            ham_lf_beta=ham_lf_beta,
            **smd_ablation_kwargs,
            ) for idx, i in enumerate(spectrallayer_num)])  # NOTE: ablation switches

        # NOTE: LD removed; no spaspe_attention_head is created.

    def forward(self, input):
        features, spe_features = [], []
        spa_input = input
        spe_input = input.clone()
        spa_input = rearrange(spa_input, 'b (g c1) h w -> (b g) c1 h w', g=self.bands_group)

        x1, x2 = spa_input, spe_input[:, None]  # b c s h w
        B = x2.shape[0]
        merge_position = 0

        for idx, encoder in enumerate(self.encoder_stages):
            x1 = encoder(x1)

            # NOTE: LD removed; SK fusion (option 1) on stage5.
            if idx == len(self.encoder_stages) - 1:
                x1_grouped = rearrange(x1, '(b g) c h w -> b g c h w', b=B, g=self.bands_group)
                ensemble_feature = self.sk_fusion(x1_grouped)
            else:
                ensemble_feature = reduce(x1, '(b g) c h w -> b c h w', 'mean', b=B)
            features.append(ensemble_feature)
            if self.linkpos[idx]:
                x2 = self.spe_encoder_stages[merge_position](x2)
                x1, x2 = self.merge_stages[merge_position](x1, x2)
                merge_position = merge_position + 1
        return features


class SST_Seg_Dual(nn.Module):
    def __init__(self, spectral_channels, out_channels, linkpos=[0, 0, 1, 0, 1],
                 backbone='resnet34', spectral_hidden_feature=64, spatial_pretrain=False,
                 activation='sigmoid', decode_choice='unet',
                 bands_group=1, spe_kernel_size=1, merge_spe_downsample=[2, 1],
                 spa_reduction=[4, 4], hw=[64, 64], rank=4,
                 tv_reg_weight=0.0, ham_lf_beta=0.1,
                 smd_ablation_kwargs=None, sk_reduction=16):  # NOTE: LD removed; ablation switches added
        super(SST_Seg_Dual, self).__init__()
        decoder_channels = (256, 128, 64, 32, 16)
        assert spectral_channels % bands_group == 0
        spatial_channels = spectral_channels // bands_group
        self.backbone = backbone
        print(f"choose backbone is {backbone} decode_choice is {decode_choice}")
        self.encoder = SpatialSpetralMixStream(spectral_channels=spectral_channels,
                                                 spectral_hidden_feature=spectral_hidden_feature,
                                                 spatial_pretrain=spatial_pretrain,
                                                 backbone=backbone,
                                                 linkpos=linkpos,
                                                 bands_group=bands_group,
                                                 spe_kernel_size=spe_kernel_size,
                                                 spa_reduction=spa_reduction,
                                                 merge_spe_downsample=merge_spe_downsample,
                                                 decode_choice=decode_choice,
                                                 hw=hw,
                                                 rank=rank,
                                                 tv_reg_weight=tv_reg_weight,  # NOTE: LD removed; no attention_group
                                                 ham_lf_beta=ham_lf_beta,
                                                 smd_ablation_kwargs=smd_ablation_kwargs,
                                                 sk_reduction=sk_reduction)  # NOTE: ablation switches


        if decode_choice == 'unet':
            self.decoder = decoders.unet.decoder.UnetDecoder(
                encoder_channels=list([spatial_channels] + self.encoder.spa_feature_dim),
                decoder_channels=decoder_channels,
                n_blocks=5)
        elif decode_choice == 'fpn':
            self.decoder = decoders.fpn.decoder.FPNDecoder(
                encoder_channels=list([spatial_channels] + self.encoder.spa_feature_dim))
        elif decode_choice == 'unetplusplus':
            self.decoder = decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
                encoder_channels=list([spatial_channels] + self.encoder.spa_feature_dim),
                decoder_channels=decoder_channels)
        elif decode_choice == 'deeplabv3plus':
            self.decoder = decoders.deeplabv3.decoder.DeepLabV3PlusDecoder(
                encoder_channels=list([spatial_channels] + self.encoder.spa_feature_dim),
                output_stride=16)

        if decode_choice == 'unet' or decode_choice == 'unetplusplus':
            upsampling = 1
        else:
            upsampling = 4

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1] if decode_choice == 'unet' or decode_choice == 'unetplusplus' else self.decoder.out_channels,
            out_channels=out_channels,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling)

    def forward(self, input):
        features = self.encoder(input)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks
