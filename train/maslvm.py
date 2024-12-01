import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
import logging
from KAN.kan_layer import KANLayer
from KANchannel import KANChannelAttention
from KANself import KANSelfAttention
from MultiKANattention import KANMultiEncoderAttention
from PIL import Image

from timm.models.layers import trunc_normal_, DropPath
from functools import partial
import pytorch_lightning as pl
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from typing import Optional, Sequence, Union
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock


class TwoConv(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

class Down(nn.Sequential):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        feat :int = 96,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

class filter_trans(nn.Module):
    def __init__(self, mode='low', in_channels=3, feat=32):
        super(filter_trans, self).__init__()
        self.mode = mode
        self.feat = feat
        self.channel_conv = nn.Conv2d(in_channels, feat, kernel_size=1)  # 调整通道数

    def forward(self, x):
        f = torch.fft.fftn(x, dim=(2, 3))
        fshift = torch.fft.fftshift(f)
        if self.mode == 'high':
            fshift = torch.fft.fftshift(f)
        
        fshift_real = torch.real(fshift)  # 转换为实数
        fshift_real = self.channel_conv(fshift_real)  # 调整通道数
        
        return fshift_real

class FINE(nn.Module):
    def __init__(self, rate, feat):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(rate), requires_grad=True)
        self.feat = feat
        self.mask = nn.Parameter(torch.ones(1, feat, 1, 1), requires_grad=True)  # Ensure mask dimension matches feature channels
        self.channel_conv = nn.Conv2d(feat, feat, kernel_size=1)  # Add a convolution layer to match channels

    def forward(self, x, fier):
        # Interpolate fier to match the dimensions of x
        fier = F.interpolate(fier, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Ensure fier has the same number of channels as self.feat
        if fier.shape[1] != self.feat:
            raise ValueError(f"Expected fier to have {self.feat} channels, but got {fier.shape[1]}")

        # Convert fier to real values before applying the convolution
        fier_real = torch.real(fier)
        fier_real = self.channel_conv(fier_real)

        fier2 = fier_real * self.mask
        x_fft = torch.fft.fftn(x, dim=(2, 3))
        x_fft = torch.fft.fftshift(x_fft)
        y = x_fft * self.rate + fier2 * (1 - self.rate)
        y = torch.fft.fftshift(y)
        y_ifft = torch.fft.ifftn(y, dim=(2, 3))
        y_ifft_real = y_ifft.real

        return y_ifft_real
      
class UpCat(nn.Module):
    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x   
    
class FFT(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,
        out_channels: int = 2,
        features: Sequence[int] = (32, 64, 128, 256, 512, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        depths=[2, 2, 2, 2],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 512,
        conv_block: bool = True,
        res_block: bool = True,
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout, feat=96)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout, feat=48)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout, feat=24)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout, feat=12)

        # 添加 KAN 通道注意力机制
        self.kan_attention1 = KANChannelAttention(in_channels=fea[1])
        self.kan_attention2 = KANChannelAttention(in_channels=fea[2])
        self.kan_attention3 = KANChannelAttention(in_channels=fea[3])
        self.kan_attention4 = KANChannelAttention(in_channels=fea[4])

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)

        self.fine1 = FINE(0.5, 32)
        self.fine2 = FINE(0.5, 64)
        self.fine3 = FINE(0.5, 128)
        self.fine4 = FINE(0.5, 256)
        self.fine5 = FINE(0.5, 512)

        self.filter_trans1 = filter_trans('low', in_channels=in_channels, feat=32)  # 设置 in_channels 和 feat 参数
        self.filter_trans2 = filter_trans('low', in_channels=32, feat=64)  # 设置 in_channels 和 feat 参数
        self.filter_trans3 = filter_trans('low', in_channels=64, feat=128)  # 设置 in_channels 和 feat 参数
        self.filter_trans4 = filter_trans('low', in_channels=128, feat=256)  # 设置 in_channels 和 feat 参数
        self.filter_trans5 = filter_trans('low', in_channels=256, feat=512)  # 设置 in_channels 和 feat 参数

    def forward(self, x: torch.Tensor):
        filter_low1 = self.filter_trans1(x)
        x0 = self.conv_0(x)
        x0 = self.fine1(x0, filter_low1) * x0
        # print(x0.shape)

        filter_low2 = self.filter_trans2(x0)
        x1 = self.down_1(x0)
        x1 = self.kan_attention1(x1)
        x1 = self.fine2(x1, filter_low2) * x1
        # print(x1.shape)
        
        filter_low3 = self.filter_trans3(x1)
        x2 = self.down_2(x1)
        x2 = self.kan_attention2(x2)
        x2 = self.fine3(x2, filter_low3) * x2
        # print(x2.shape)
        
        filter_low4 = self.filter_trans4(x2)
        x3 = self.down_3(x2)
        x3 = self.kan_attention3(x3)
        x3 = self.fine4(x3, filter_low4) * x3
        # print(x3.shape)
        
        filter_low5 = self.filter_trans5(x3)
        x4 = self.down_4(x3)
        x4 = self.kan_attention4(x4)
        x4 = self.fine5(x4, filter_low5) * x4
        # print(x4.shape)

        u4 = self.upcat_4(x4, x3)
        # print(u4.shape)

        u3 = self.upcat_3(u4, x2)
        # print(u3.shape)

        return u3

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3PlusDeepSup(nn.Module):
    def __init__(self):
        super(UNet3PlusDeepSup, self).__init__()
        self.image_channels = 3
        self.filters = [64, 128, 256, 512, 1024]
        self.input_layer = nn.Identity()

        # Encoder                                               
        # block 1
        self.e1 = ConvBlock(self.image_channels, self.filters[0])       # 320*320*64
        # block 2
        self.pool_e2 = nn.MaxPool2d(kernel_size=(2, 2))         # 160*160*64
        self.e2 = ConvBlock(self.filters[0], self.filters[1])   # 160*160*128
        # block 3
        self.pool_e3 = nn.MaxPool2d(kernel_size=(2, 2))         # 80*80*128
        self.e3 = ConvBlock(self.filters[1], self.filters[2])   # 80*80*256
        # block 4
        self.pool_e4 = nn.MaxPool2d(kernel_size=(2, 2))         # 40*40*256
        self.e4 = ConvBlock(self.filters[2], self.filters[3])   # 40*40*512
        # block 5
        self.pool_e5 = nn.MaxPool2d(kernel_size=(2, 2))         # 20*20*512
        self.e5 = ConvBlock(self.filters[3], self.filters[4])   # 20*20*1024

        # self.kan_attention = KANSelfAttention(in_channels=self.filters[4])


    def forward(self, x):
        # Encode
        x = self.input_layer(x)
        e1 = self.e1(x)

        e2 = self.pool_e2(e1)
        e2 = self.e2(e2)

        e3 = self.pool_e3(e2)
        e3 = self.e3(e3)

        e4 = self.pool_e4(e3)
        e4 = self.e4(e4)

        e5 = self.pool_e5(e4)
        e5 = self.e5(e5)
        # e5 = self.kan_attention(e5)

        return e5

class DeformableLaplacian(nn.Module):
    def __init__(self, in_channels):
        super(DeformableLaplacian, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        offsets = self.conv1(x)
        deformable_kernel = self.conv2(x + offsets)
        return x + deformable_kernel

class BOD(nn.Module):
    def __init__(self, in_channels=3):
        super(BOD, self).__init__()
        self.backbone = UNet3PlusDeepSup()
        self.deformable_laplacian = DeformableLaplacian(1024)

    def forward(self, x):
        features = self.backbone(x)
        refined_features = self.deformable_laplacian(features)
        return refined_features
       
class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=32, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(256),    
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(256),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo
       
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(transformer_dim=256,
                         transformer=TwoWayTransformer(
                             depth=2,
                             embedding_dim=256,
                             mlp_dim=2048,
                             num_heads=8,
                         ),
                         num_multimask_outputs=3,
                         activation=nn.GELU,
                         iou_head_depth=3,
                         iou_head_hidden_dim=256,)

        assert model_type in ["vit_b", "vit_l", "vit_h"]

        checkpoint_dict = {"vit_b": "pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
                           "vit_l": "pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           'vit_h': "pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        for n, p in self.named_parameters():
            p.requires_grad = False

        transformer_dim = 256
        vit_dim_dict = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat1 = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2))

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
        
        self.compress_vit_feat_with_PixelShuffle = nn.Sequential(
                                nn.Conv2d(vit_dim, transformer_dim*2, kernel_size=9, stride=1, padding=4), # in_channel: transformer_dim, out_channel: transformer_dim 
                                nn.PixelShuffle(2),  # Upsample by a scale_factor=2, channel = channel_dim/4
                                LayerNorm2d(transformer_dim // 2),
                                nn.GELU(),
                                nn.Conv2d(transformer_dim // 2, transformer_dim // 2, kernel_size=9, stride=1, padding=4), 
                                nn.PixelShuffle(2), # Upsample by a scale_factor=2, channel = channel_dim/4
                                )
        
        self.aff = iAFF()
        self.conv = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1)
        self.kan_multi_attention = KANMultiEncoderAttention(in_channels=32, num_encoders=3)
    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            hq_token_only: bool,
            interm_embeddings: torch.Tensor,
            ori_image: torch.Tensor,
            fft_features: torch.Tensor,  
            bod_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vit_features_out1 = self.compress_vit_feat1(interm_embeddings[0].permute(0, 3, 1, 2))
        hq_features = self.embedding_encoder(image_embeddings) + vit_features_out1 
        bod_features = self.compress_vit_feat_with_PixelShuffle(bod_features)
        fft_features = self.conv(fft_features)
        combine_features = self.kan_multi_attention(hq_features,bod_features,fft_features)
        hq_features = self.aff(hq_features,combine_features)
        hq_features = self.aff(hq_features,fft_features)
        hq_features = self.aff(hq_features,bod_features)
        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature=hq_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks, 0)
        iou_preds = torch.cat(iou_preds, 0)

        if multimask_output:
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            mask_slice = slice(0, 1)
            masks_sam = masks[:, mask_slice]

        masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens), :, :]

        if hq_token_only:
            return masks_hq
        else:
            return masks_sam, masks_hq

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape


        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_anns_mask(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious,name):
    if len(masks) == 0:
        return
    if not os.path.exists (filename): 
        os.mkdir (filename)
    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10,10))

        mask_image = show_mask(mask, plt.gca())
        mask_image = np.asarray(mask_image)
        mask_image = np.squeeze(mask_image)  # Removes singleton dimensions

        mask_image = (mask_image * 255).astype(np.uint8)  # Converts to uint8

        save_dir = os.path.join(filename,name[0]) + ".png"
        Image.fromarray(mask_image).convert('L').save(save_dir)


        plt.axis('off')
        plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]

    mask_image = mask.reshape(h, w, 1)
    return mask_image
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=12, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', type=int, help='local rank for dist')
    parser.add_argument('--distributed', action='store_true')

    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize',default=False, action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the sam_decoder training checkpoint for evaluation")
    parser.add_argument("--restore-fft", type=str,
                        help="The path to the FFT model checkpoint for evaluation")
    parser.add_argument("--restore-bod", type=str,
                        help="The path to the BOD model checkpoint for evaluation")
    return parser.parse_args()

def main(net, fft, bod, train_datasets, valid_datasets, args):
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')
    if (args.distributed):
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        args.gpu = torch.device("cuda", local_rank)
        print('rank: {}'.format(args.rank))
        print('local_rank: {}'.format(args.local_rank))
        print('world size: {}'.format(args.world_size))
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.output,'example.log'), level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter()
                                                                    ],
                                                        batch_size = args.batch_size_train,
                                                        training = True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    if torch.cuda.is_available():
        net.cuda()
        fft.cuda()
        bod.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    net_without_ddp = net.module
    fft = torch.nn.parallel.DistributedDataParallel(fft, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    fft_without_ddp = fft.module
    bod = torch.nn.parallel.DistributedDataParallel(bod, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    bod_without_ddp = bod.module
    ### --- Step 3: Train or Evaluate ---
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch
        
        # Optimizer and Scheduler for FFT
        optimizer_fft = optim.Adam(fft_without_ddp.parameters(), lr=args.learning_rate*100, weight_decay=1e-8)
        lr_scheduler_fft = torch.optim.lr_scheduler.StepLR(optimizer_fft, args.lr_drop_epoch)
        lr_scheduler_fft.last_epoch = args.start_epoch
        
        # Optimizer and Scheduler for BOD
        optimizer_bod = optim.Adam(bod_without_ddp.parameters(), lr=args.learning_rate*100, weight_decay=1e-8)
        lr_scheduler_bod = torch.optim.lr_scheduler.StepLR(optimizer_bod, args.lr_drop_epoch)
        lr_scheduler_bod.last_epoch = args.start_epoch

        train(args, net, fft, bod, optimizer, optimizer_fft, optimizer_bod, train_dataloaders, valid_dataloaders, lr_scheduler, lr_scheduler_fft, lr_scheduler_bod)
    else:
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        _ = sam.to(device=args.device)
        sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)

        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net_without_ddp.load_state_dict(torch.load(args.restore_model))
            else:
                net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))

        if args.restore_fft:
            print("restore FFT model from:", args.restore_fft)
            if torch.cuda.is_available():
                fft_without_ddp.load_state_dict(torch.load(args.restore_fft))
            else:
                fft_without_ddp.load_state_dict(torch.load(args.restore_fft, map_location="cpu"))

        if args.restore_bod:
            print("restore BOD model from:", args.restore_bod)
            if torch.cuda.is_available():
                bod_without_ddp.load_state_dict(torch.load(args.restore_bod))
            else:
                bod_without_ddp.load_state_dict(torch.load(args.restore_bod, map_location="cpu"))
        
        evaluate(args, net, fft, bod, sam, valid_dataloaders, args.visualize)


def train(args, net, fft, bod, optimizer, optimizer_fft, optimizer_bod, train_dataloaders, valid_dataloaders, lr_scheduler, lr_scheduler_fft, lr_scheduler_bod):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    net.train()
    _ = net.to(device=args.device)

    fft.train()
    _ = fft.to(device=args.device)

    bod.train()
    _ = bod.to(device=args.device)

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    
    for epoch in range(epoch_start, epoch_num): 
        print("epoch:   ", epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")

        for data in metric_logger.log_every(train_dataloaders, 100):
            inputs, labels = data['image'], data['label']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # input prompt
            input_keys = ['box', 'point', 'noise_mask']
            labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
            try:
                labels_points = misc.masks_sample_points(labels[:, 0, :, :])
            except:
                input_keys = ['box', 'noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            fft_features = fft(torch.stack([batched_input[i_l]['image'] for i_l in range(batch_len)], dim=0).float())
            bod_features = bod(torch.stack([batched_input[i_l]['image'] for i_l in range(batch_len)], dim=0).float())

            masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
                ori_image=inputs,
                fft_features=fft_features,
                bod_features=bod_features,
            )

            loss_mask, loss_dice = loss_masks(masks_hq, labels / 255.0, len(masks_hq))
            loss = loss_mask + loss_dice
            loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer_fft.zero_grad()
            optimizer_bod.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_bod.step()
            optimizer_fft.step()
            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)


        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        lr_scheduler_fft.step()
        lr_scheduler_bod.step()
       
        test_stats = evaluate(args, net, fft, bod, sam, valid_dataloaders,epoch=epoch)
        train_stats.update(test_stats)
        
        net.train()  
        fft.train()
        bod.train()

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            model_name_fft = "/fft_epoch_"+str(epoch)+".pth"
            model_name_bod = "/bod_epoch_"+str(epoch)+".pth"

            print('come here save at', args.output + model_name)
            misc.save_on_master(net.module.state_dict(), args.output + model_name)
            misc.save_on_master(fft.module.state_dict(), args.output + model_name_fft)
            misc.save_on_master(bod.module.state_dict(), args.output + model_name_bod)

    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if misc.is_main_process():
        sam_ckpt = torch.load(args.checkpoint)
        hq_decoder = torch.load(args.output + model_name)
        for key in hq_decoder.keys():
            sam_key = 'mask_decoder.' + key
            if sam_key not in sam_ckpt.keys():
                sam_ckpt[sam_key] = hq_decoder[key]
        model_name = "/sam_hq_epoch_" + str(epoch) + ".pth"
        torch.save(sam_ckpt, args.output + model_name)

def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    
    # print(len(preds))
    # print("Average IOU:",iou / len(preds))
    return iou / len(preds)

def compute_dice(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    dice = 0
    for i in range(0,len(preds)):
        dice = dice + misc.mask_dice(postprocess_preds[i],target[i])

    # print("Average Dice:",dice / len(preds))
    return dice / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    # print(torch.unique(postprocess_preds))
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    # print("Average boundary_iou:",iou / len(preds))
    return iou / len(preds)

def evaluate(args, net, fft, bod, sam, valid_dataloaders, visualize=False ,epoch=0):
    net.eval()  
    fft.eval()  
    bod.eval()  

    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader, 1000):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori, name = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label'], data_val['name']

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(labels_val[:, 0, :, :])
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()

                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            fft_features = fft(torch.stack([batched_input[i_l]['image'] for i_l in range(batch_len)], dim=0).float())
            bod_features = bod(torch.stack([batched_input[i_l]['image'] for i_l in range(batch_len)], dim=0).float())

            masks_sam, masks_hq = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=False,
                interm_embeddings=interm_embeddings,
                ori_image=inputs_val,
                fft_features=fft_features,
                bod_features=bod_features,
            )

            iou = compute_iou(masks_hq, labels_ori)
            dice = compute_dice(masks_hq, labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq, labels_ori)

            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(masks_hq.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()

                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    print('base:', base)
                    save_base = args.output
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns_mask(masks_hq_vis[ii], None, None, None, os.path.join(save_base,'pred_val') , imgs_ii, show_iou, show_boundary_iou,name)

            loss_dict = {"val_iou_" + str(k): iou, "val_dice_" + str(k): dice, "val_boundary_iou_" + str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)
            del inputs_val, labels_val, labels_ori, masks_sam, masks_hq  
            torch.cuda.empty_cache()  #

        print('============================')
        # gather the stats from all processes
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        metric_logger.synchronize_between_processes()
        loss_message = f"Epoch : {epoch}, Loss Dictionary: {resstat}"

        # 写入日志
        logging.info(loss_message)
        test_stats.update(resstat)

    return test_stats

if __name__ == "__main__":

    dataset_ISIC_train = {"name": "ISIC",
                "im_dir": "/home/lab2/32t/g/sam-kan/ISIC2018resize/train_A",
                "gt_dir": "/home/lab2/32t/g/sam-kan/ISIC2018resize/train_B",
                "im_ext": ".jpg",
                "gt_ext": "_segmentation.png"}
    
    dataset_ISIC_val = {"name": "ISIC",
                "im_dir": "/home/lab2/32t/g/sam-kan/ISIC2018resize/ISIC2018_Task1-2_Validation_Input",
                "gt_dir": "/home/lab2/32t/g/sam-kan/ISIC2018resize/ISIC2018_Task1_Validation_GroundTruth",
                "im_ext": ".jpg",
                "gt_ext": "_segmentation.png"}
    
    dataset_ISIC_test = {"name": "ISIC",
            "im_dir": "/home/lab2/32t/g/sam-kan/ISIC2018resize/test_A",
            "gt_dir": "/home/lab2/32t/g/sam-kan/ISIC2018resize/test_B",
            "im_ext": ".jpg",
            "gt_ext": "_segmentation.png"}

    dataset_ISIC2017_train = {"name": "ISIC2017",
                "im_dir": "/home/lab2/32t/g/sam-kan/ISIC2017/imagesTr",
                "gt_dir": "/home/lab2/32t/g/sam-kan/ISIC2017/labelsTr",
                "im_ext": ".jpg",
                "gt_ext": "_segmentation.png"}
    
    dataset_ISIC2017_test = {"name": "ISIC2017",
                "im_dir": "/home/lab2/32t/g/sam-kan/ISIC2017/imagesTs",
                "gt_dir": "/home/lab2/32t/g/sam-kan/ISIC2017/labelsTs",
                "im_ext": ".jpg",
                "gt_ext": "_segmentation.png"}
    
    dataset_ISIC2017_val = {"name": "ISIC2017",
            "im_dir": "/home/lab2/32t/g/sam-kan/ISIC2017/imagesVal",
            "gt_dir": "/home/lab2/32t/g/sam-kan/ISIC2017/labelsVal",
            "im_ext": ".jpg",
            "gt_ext": "_segmentation.png"}

    dataset_CVC_train = {"name": "CVC",
                "im_dir": "/home/lab2/32t/g/sam-kan/CVC-ClinicDB/trainA",
                "gt_dir": "/home/lab2/32t/g/sam-kan/CVC-ClinicDB/trainB",
                "im_ext": ".png",
                "gt_ext": ".png"}
    
    dataset_CVC_test = {"name": "CVC",
            "im_dir": "/home/lab2/32t/g/sam-kan/CVC-ClinicDB/testA",
            "gt_dir": "/home/lab2/32t/g/sam-kan/CVC-ClinicDB/testB",
            "im_ext": ".png",
            "gt_ext": ".png"}

    dataset_SEG_train = {"name": "SEG",
                "im_dir": "/home/lab2/32t/g/sam-kan/Kvasir-SEG/trainA",
                "gt_dir": "/home/lab2/32t/g/sam-kan/Kvasir-SEG/trainB",
                "im_ext": ".png",
                "gt_ext": ".png"}
    
    dataset_SEG_test = {"name": "SEG",
            "im_dir": "/home/lab2/32t/g/sam-kan/Kvasir-SEG/testA",
            "gt_dir": "/home/lab2/32t/g/sam-kan/Kvasir-SEG/testB",
            "im_ext": ".png",
            "gt_ext": ".png"}

    dataset_Poly_train = {"name": "Poly",
                "im_dir": "/home/lab2/32t/g/sam-kan/PolypGen2021/trainA",
                "gt_dir": "/home/lab2/32t/g/sam-kan/PolypGen2021/trainB",
                "im_ext": ".png",
                "gt_ext": ".png"}
    
    dataset_Poly_val = {"name": "Poly",
                "im_dir": "/home/lab2/32t/g/sam-kan/PolypGen2021/val_images",
                "gt_dir": "/home/lab2/32t/g/sam-kan/PolypGen2021/val_masks",
                "im_ext": ".png",
                "gt_ext": ".png"}
    
    dataset_Poly_test = {"name": "Poly",
            "im_dir": "/home/lab2/32t/g/sam-kan/PolypGen2021/test_images",
            "gt_dir": "/home/lab2/32t/g/sam-kan/PolypGen2021/test_masks",
            "im_ext": ".png",
            "gt_ext": ".png"}

    dataset_ph2_test = {"name": "ph2",
            "im_dir": "/home/lab2/32t/g/sam-kan/ph2/trainA",
            "gt_dir": "/home/lab2/32t/g/sam-kan/ph2/trainB",
            "im_ext": ".png",
            "gt_ext": "_lesion.png"}

    train_datasets = [dataset_ISIC_train, dataset_ISIC2017_train, dataset_Poly_train, dataset_SEG_train, dataset_CVC_train]
    valid_datasets = [dataset_ISIC_val, dataset_ISIC2017_val, dataset_Poly_test, dataset_SEG_test, dataset_CVC_test, dataset_ph2_test] 


    args = get_args_parser()
    
    net = MaskDecoderHQ(args.model_type) 
    fft = FFT()
    _ = fft.to(device=args.device)
    bod = BOD()
    _ = bod.to(device=args.device)

    main(net,fft,bod, train_datasets, valid_datasets, args)


