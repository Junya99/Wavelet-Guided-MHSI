#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored & Improved: Added 3 key innovations on top of basemodel_0205
  1. BandAwareAggregation  — learnable band-group weighting per encoder stage
  2. SpectralEnhancedSkip  — inject spectral features into decoder skip connections
  3. BoundaryRefinementHead — explicit boundary prediction with auxiliary supervision
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


# ============================================================
# Existing modules (unchanged from basemodel_0205)
# ============================================================

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
        b, g, c, h, w = x.shape
        pooled = x.mean(dim=(-1, -2))
        weights = self.fc2(F.relu(self.fc1(pooled)))
        weights = weights.softmax(dim=1)
        weights = weights.view(b, g, 1, 1, 1)
        return (x * weights).sum(dim=1)


class Hamburger(nn.Module):
    def __init__(self, in_c, args=None):
        super().__init__()
        ham_type = getattr(args, 'HAM_TYPE', 'NMF')
        C = in_c
        num_groups = self._pick_group_count(C)
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

    def forward(self, x):
        ham_x = self.norm(x)
        ham_x = self.lower_bread(ham_x)
        ham_x = self.ham(ham_x)
        out = F.relu(x + ham_x, inplace=True)
        return out

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)


class SMD(nn.Module):
    """
    SMD Block (unchanged from basemodel_0205).
    """
    def __init__(self, in_channels, hidden_feature, spe_reduction=1,
                 dim_reduction=4, kernel_size=1, lf_alpha_init=1.0,
                 use_multiscale=True, use_hf=False, ham_gate_init=1.0,
                 ham_group_size=0, sparse_reg_weight=0.0, orth_reg_weight=0.0,
                 tv_reg_weight=0.0, ham_lf_beta=0.1):
        super(SMD, self).__init__()
        self.spe_reduction = spe_reduction
        self.hidden_feature = hidden_feature
        self.use_multiscale = True
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
        self.spec_local_k5 = nn.Conv1d(hidden_feature, hidden_feature, kernel_size=5, padding=2, groups=hidden_feature)
        self.spec_local_k7 = nn.Conv1d(hidden_feature, hidden_feature, kernel_size=7, padding=3, groups=hidden_feature)
        self.qkv = nn.Linear(hidden_feature, hidden_feature * 3)
        self.ham = Hamburger(hidden_feature)
        self.ham_gate = nn.Parameter(torch.tensor(float(ham_gate_init)))
        if self.ham_group_size and self.ham_group_size > 0:
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
        x_local = x_local + self.spec_local_k5(x_base) + self.spec_local_k7(x_base)

        qkv = self.qkv(x_local.transpose(1, 2))
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        context = torch.matmul(k.transpose(-2, -1), v)
        x_attn = torch.matmul(q, context).transpose(1, 2)

        x_wave = x_base
        x_lf = self.wavelet_transform(x_wave)
        x_lf = F.interpolate(x_lf, size=s, mode='linear', align_corners=False)

        stat_feat = torch.cat(
            [x_base.mean(-1), x_base.std(-1, unbiased=False), x_lf.mean(-1)],
            dim=-1
        )
        alpha = self.lf_gate(stat_feat).unsqueeze(-1)
        beta = torch.sigmoid(self.ham_lf_beta_raw) * 0.2
        ham_in = x_local + x_attn + (beta * alpha * x_lf)

        if self.ham_group_size and self.ham_group_size > 0:
            chunks = torch.split(ham_in, self.ham_group_size, dim=-1)
            ham_chunks = [self.ham(chunk) for chunk in chunks]
            x_ham = torch.cat(ham_chunks, dim=-1)
            x_ham = self.ham_fusion(x_ham)
        else:
            x_ham = self.ham(ham_in)

        x_ham = self.ham_gate * x_ham
        self._last_ham_output = x_ham
        out = x_base + x_ham + alpha * x_lf
        out = rearrange(out, '(b h w) c s -> b c s h w', b=b, h=h_, w=w_)
        return out

    def regularization_loss(self):
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

    def forward(self, spa_input, spe_input):
        spa_input = rearrange(spa_input, '(b g) c h w -> b g c h w', g=self.bands_group)
        spe2spa_input = reduce(spe_input, 'b c (s1 p) h w -> b s1 c h w', 'mean',
                               p=self.merge_spe_downsample)
        spa_input = self.merge_conv_spa(
            rearrange(torch.cat((spa_input, spe2spa_input), 2), 'b g c h w -> (b g) c h w'))
        return spa_input, spe_input


# ============================================================
# Innovation 1: Band-Aware Aggregation (BAA)
# ============================================================
class BandAwareAggregation(nn.Module):
    """
    Replaces simple mean aggregation of band-group features.

    Motivation: In multispectral imaging, different band groups capture
    different tissue properties (e.g., hemoglobin absorption, water content).
    Simple averaging treats all band groups equally, discarding the fact that
    certain bands are more informative at certain spatial locations.

    This module learns per-pixel, per-group importance weights via a lightweight
    channel-spatial attention mechanism, enabling adaptive band-group fusion.

    The attention weights are interpretable: they reveal which spectral bands
    contribute most to the segmentation at each spatial position.
    """
    def __init__(self, channels, num_groups, reduction=8):
        super().__init__()
        self.num_groups = num_groups
        hidden = max(channels // reduction, 4)

        # Channel attention: learn per-group importance from global context
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B*G, C, 1, 1)
            nn.Flatten(),              # (B*G, C)
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

        # Spatial attention: capture spatial-varying band importance
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

        # Final projection to refine aggregated features
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(self._pick_group_count(channels), channels),
            nn.GELU(),
        )

    @staticmethod
    def _pick_group_count(channels):
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x, B):
        """
        Args:
            x: (B*G, C, H, W) — features from all band groups
            B: batch size
        Returns:
            out: (B, C, H, W) — aggregated features
        """
        G = self.num_groups
        _, C, H, W = x.shape

        # Channel attention per group
        ca = self.channel_attn(x)  # (B*G, C)
        ca = rearrange(ca, '(b g) c -> b g c', b=B, g=G)

        # Spatial attention per group
        sa = self.spatial_attn(x)  # (B*G, 1, H, W)
        sa = rearrange(sa, '(b g) 1 h w -> b g 1 h w', b=B, g=G)

        # Combine channel + spatial attention as importance weight
        ca = ca.unsqueeze(-1).unsqueeze(-1)  # (B, G, C, 1, 1)
        weight = ca + sa  # broadcast: (B, G, C, H, W)
        weight = weight.softmax(dim=1)  # normalize across groups

        # Weighted aggregation
        x_grouped = rearrange(x, '(b g) c h w -> b g c h w', b=B, g=G)
        out = (x_grouped * weight).sum(dim=1)  # (B, C, H, W)

        # Refinement
        out = self.proj(out)
        return out


# ============================================================
# Innovation 2: Spectral-Enhanced Skip Connection (SESC)
# ============================================================
class SpectralEnhancedSkip(nn.Module):
    """
    Injects spectral-stream features into decoder skip connections.

    Motivation: In the original model, spectral features (from SMD modules)
    are only used to refine the spatial encoder via merge_block, then discarded.
    The decoder receives no spectral context, losing tissue-specific spectral
    signatures that could disambiguate visually similar but spectrally distinct
    regions (e.g., healthy vs. inflamed tissue with similar appearance but
    different spectral responses).

    This module projects spectral features to match the spatial feature space,
    then uses a learned gating mechanism to selectively enhance skip features
    with spectral context — only where spectral information is beneficial.
    """
    def __init__(self, spa_channels, spe_channels):
        super().__init__()
        num_groups = self._pick_group_count(spa_channels)

        # Project spectral features to spatial feature space
        self.spe_proj = nn.Sequential(
            nn.Conv2d(spe_channels, spa_channels, kernel_size=1),
            nn.GroupNorm(num_groups, spa_channels),
            nn.GELU(),
        )

        # Gating: decide how much spectral info to inject per pixel
        self.gate = nn.Sequential(
            nn.Conv2d(spa_channels * 2, spa_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Final refinement after injection
        self.refine = nn.Sequential(
            nn.Conv2d(spa_channels, spa_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, spa_channels),
            nn.GELU(),
        )

    @staticmethod
    def _pick_group_count(channels):
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, spa_feat, spe_feat):
        """
        Args:
            spa_feat: (B, C_spa, H, W) — encoder spatial feature for skip
            spe_feat: (B, C_spe, S, H', W') — spectral feature from SMD
        Returns:
            enhanced: (B, C_spa, H, W) — spectrally-enhanced skip feature
        """
        # Pool over spectral dimension
        spe_pooled = spe_feat.mean(dim=2)  # (B, C_spe, H', W')
        # Resize to match spatial feature resolution
        spe_pooled = F.interpolate(spe_pooled, size=spa_feat.shape[-2:],
                                   mode='bilinear', align_corners=False)
        spe_proj = self.spe_proj(spe_pooled)  # (B, C_spa, H, W)

        # Learned gate: how much spectral info to use at each position
        gate = self.gate(torch.cat([spa_feat, spe_proj], dim=1))
        enhanced = spa_feat + gate * spe_proj

        return self.refine(enhanced)


# ============================================================
# Innovation 3: Boundary Refinement Head (BRH)
# ============================================================
class BoundaryRefinementHead(nn.Module):
    """
    Produces an explicit boundary prediction from the decoder output.

    Motivation: In multispectral medical image segmentation, lesion boundaries
    are often ambiguous — different spectral bands may suggest slightly different
    boundary locations. Standard decoders optimize only region-level metrics
    (Dice/BCE), ignoring boundary quality. Since Hausdorff distance is a key
    evaluation metric, explicitly predicting and supervising boundaries provides:
    1. Direct optimization signal for boundary quality
    2. An interpretable edge map showing model's boundary confidence
    3. Gradient flow that emphasizes hard boundary pixels

    The module extracts edge features via a lightweight conv branch and produces
    a single-channel boundary probability map for auxiliary supervision.
    """
    def __init__(self, in_channels):
        super().__init__()
        mid_ch = max(in_channels // 4, 8)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=3, padding=1),
            nn.GroupNorm(self._pick_group_count(mid_ch), mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.GroupNorm(self._pick_group_count(mid_ch), mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _pick_group_count(channels):
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — decoder output feature
        Returns:
            edge_map: (B, 1, H, W) — boundary probability map
        """
        return self.edge_conv(x)


# ============================================================
# Modified Encoder: SpatialSpetralMixStream
# ============================================================
class SpatialSpetralMixStream(nn.Module):
    def __init__(self, spectral_channels, linkpos=[0, 0, 1, 0, 1], bands_group=15,
                 backbone='resnet34', spectral_hidden_feature=32,
                 spatial_pretrain=False, spe_kernel_size=1,
                 merge_spe_downsample=[2, 1],
                 spa_reduction=[4, 4], decode_choice='unet',
                 tv_reg_weight=0.0, ham_lf_beta=0.1,
                 hw=[64, 64], rank=4, sk_reduction=16):

        super(SpatialSpetralMixStream, self).__init__()
        self.spectral_channels = spectral_channels
        self.linkpos = linkpos
        self.spe_feature_dim = [1, spectral_hidden_feature * 1, spectral_hidden_feature * 2,
                                spectral_hidden_feature * 4,
                                spectral_hidden_feature * 8, spectral_hidden_feature * 16]
        self.bands_group = bands_group

        spectrallayer_num = np.array(linkpos).nonzero()[0]
        self.spatial_backbone = get_encoder(name=backbone,
                                            in_channels=spectral_channels // bands_group,
                                            depth=5, weights='imagenet' if spatial_pretrain else None,
                                            output_stride=16 if decode_choice == 'deeplabv3plus' else 32)
        self.spa_feature_dim = list(self.spatial_backbone.out_channels[1:])
        self.encoder_stages = self.spatial_backbone.get_stages()
        # Full per-stage output channels: [in_ch, 64, 64, 128, 256, 512]
        self.all_stage_channels = list(self.spatial_backbone.out_channels)

        # [CHANGED] SKFusion only for the last stage (unchanged)
        self.sk_fusion = SKFusion(self.spa_feature_dim[-1], reduction=sk_reduction)

        # [NEW - Innovation 1] Band-Aware Aggregation for non-last stages
        # Replace simple mean with learnable aggregation
        # Note: encoder_stages has len(all_stage_channels) stages;
        #       all but the last use BandAwareAggregation.
        num_stages = len(self.all_stage_channels)
        self.band_aggregators = nn.ModuleList([
            BandAwareAggregation(channels=self.all_stage_channels[i], num_groups=bands_group)
            for i in range(num_stages - 1)  # all except last stage
        ])

        self.merge_stages = nn.ModuleList([merge_block(self.spa_feature_dim[i - 1], self.spe_feature_dim[idx + 1],
                                                       bands_group=bands_group,
                                                       merge_spe_downsample=merge_spe_downsample[idx])
                                           for idx, i in enumerate(spectrallayer_num)])

        self.spe_encoder_stages = nn.ModuleList([SMD(
            self.spe_feature_dim[idx], self.spe_feature_dim[idx + 1],
            spa_reduction[idx],
            kernel_size=spe_kernel_size,
            use_multiscale=True,
            tv_reg_weight=tv_reg_weight,
            ham_lf_beta=ham_lf_beta
            ) for idx, i in enumerate(spectrallayer_num)])

        # [NEW - Innovation 2] Spectral-Enhanced Skip Connections
        # Create one SESC for each merge position (where we have spectral features)
        self.spe_skip_enhancers = nn.ModuleList([
            SpectralEnhancedSkip(
                spa_channels=self.spa_feature_dim[i - 1],
                spe_channels=self.spe_feature_dim[idx + 1]
            ) for idx, i in enumerate(spectrallayer_num)
        ])

    def forward(self, input):
        features, spe_features_cache = [], {}
        spa_input = input
        spe_input = input.clone()
        spa_input = rearrange(spa_input, 'b (g c1) h w -> (b g) c1 h w', g=self.bands_group)

        x1, x2 = spa_input, spe_input[:, None]  # b c s h w
        B = x2.shape[0]
        merge_position = 0

        for idx, encoder in enumerate(self.encoder_stages):
            x1 = encoder(x1)

            if idx == len(self.encoder_stages) - 1:
                # Last stage: use SKFusion (unchanged)
                x1_grouped = rearrange(x1, '(b g) c h w -> b g c h w', b=B, g=self.bands_group)
                ensemble_feature = self.sk_fusion(x1_grouped)
            else:
                # [CHANGED - Innovation 1] Use BandAwareAggregation instead of mean
                ensemble_feature = self.band_aggregators[idx](x1, B)

            features.append(ensemble_feature)

            if self.linkpos[idx]:
                x2 = self.spe_encoder_stages[merge_position](x2)
                x1, x2 = self.merge_stages[merge_position](x1, x2)

                # [NEW - Innovation 2] Cache spectral features and enhance skip features
                spe_features_cache[idx] = (merge_position, x2.clone())
                # Enhance the corresponding encoder feature with spectral info
                features[-1] = self.spe_skip_enhancers[merge_position](
                    features[-1], x2
                )

                merge_position = merge_position + 1

        return features


# ============================================================
# Modified Main Model: SST_Seg_Dual
# ============================================================
class SST_Seg_Dual(nn.Module):
    def __init__(self, spectral_channels, out_channels, linkpos=[0, 0, 1, 0, 1],
                 backbone='resnet34', spectral_hidden_feature=64, spatial_pretrain=False,
                 activation='sigmoid', decode_choice='unet',
                 bands_group=1, spe_kernel_size=1, merge_spe_downsample=[2, 1],
                 spa_reduction=[4, 4], hw=[64, 64], rank=4,
                 tv_reg_weight=0.0, ham_lf_beta=0.1,
                 sk_reduction=16):
        super(SST_Seg_Dual, self).__init__()
        decoder_channels = (256, 128, 64, 32, 16)
        assert spectral_channels % bands_group == 0
        spatial_channels = spectral_channels // bands_group
        self.backbone = backbone
        print(f"choose backbone is {backbone} decode_choice is {decode_choice}")
        print("Using Innovations: BandAwareAggregation + SpectralEnhancedSkip + BoundaryRefinement")

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
                                                 tv_reg_weight=tv_reg_weight,
                                                 ham_lf_beta=ham_lf_beta,
                                                 sk_reduction=sk_reduction)

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

        decoder_out_ch = decoder_channels[-1] if decode_choice in ('unet', 'unetplusplus') else self.decoder.out_channels

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_out_ch,
            out_channels=out_channels,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling)

        # [NEW - Innovation 3] Boundary Refinement Head
        self.boundary_head = BoundaryRefinementHead(in_channels=decoder_out_ch)
        self._upsampling = upsampling

    def forward(self, input):
        features = self.encoder(input)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        if self.training:
            # Boundary prediction (at decoder resolution, then upsample to match mask)
            edge_pred = self.boundary_head(decoder_output)
            if self._upsampling > 1:
                edge_pred = F.interpolate(edge_pred, scale_factor=self._upsampling,
                                          mode='bilinear', align_corners=False)
            return masks, edge_pred

        return masks
