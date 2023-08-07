import math
from typing import List

import torch
from pydantic import StrictBool, StrictInt
from torch import nn
from torch.nn import functional as F

from units.models.transformer import MLP, init_weights


class PositionalEncoding(nn.Module):
    def __init__(self, dim, temperature=10000, normalize=False):
        super().__init__()

        self.dim = dim
        self.normalize = normalize

        self.scale = 2 * math.pi
        freq = torch.arange(self.dim, dtype=torch.float32)
        freq = temperature ** (2 * _floor_div(freq, 2) / _floor_div(dim, 2))
        self.register_buffer("freq", freq)

    def forward(self, mask):
        not_mask = ~mask
        y_embed = torch.arange(
            1, mask.shape[1] + 1, dtype=torch.float32, device=mask.device
        ).view(1, -1, 1)
        x_embed = torch.arange(
            1, mask.shape[2] + 1, dtype=torch.float32, device=mask.device
        ).view(1, 1, -1)

        y_embed = y_embed * not_mask
        x_embed = x_embed * not_mask

        if self.normalize:
            eps = 1e-6
            y_embed = (
                (y_embed - 0.5)
                / (y_embed.max(1, keepdim=True).values + eps)
                * self.scale
            )
            x_embed = (
                (x_embed - 0.5)
                / (x_embed.max(2, keepdim=True).values + eps)
                * self.scale
            )

        pos_x = x_embed[:, :, :, None] / self.freq
        pos_y = y_embed[:, :, :, None] / self.freq

        pos_x = torch.stack(
            (torch.sin(pos_x[:, :, :, 0::2]), torch.cos(pos_x[:, :, :, 1::2])), 4
        ).flatten(3)
        pos_y = torch.stack(
            (torch.sin(pos_y[:, :, :, 0::2]), torch.cos(pos_y[:, :, :, 1::2])), 4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


def _floor_div(a, b):
    return torch.div(a, b, rounding_mode="floor")


class HybridTransformer(nn.Module):
    """
    HybridTransformer: backbone + Transformer encoder
    Args:
        backbone (Tuple): (Backbone Model, backbone_out_feature_dimensions)
        strides (List[int]): Backbone feature selection via strides.
            i.e. if strides == [8, 16],
            two backbone features which has stride [8, 16] is selected.
        dim (int): hidden dimension
        encoder (nn.Module): encoder
    """

    def __init__(
        self,
        backbone: nn.Module,
        strides: List[StrictInt],
        dim: StrictInt,
        encoder: nn.Module,
    ):
        super().__init__()

        self.backbone, self.backbone_dims = backbone
        feat_ids = [int(math.log(stride, 2)) - 1 for stride in strides]
        self.feat_ids = feat_ids

        self.input_proj = nn.ModuleList()
        for feat_id in feat_ids:
            self.input_proj.append(
                nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(self.backbone_dims[feat_id], dim),
                    nn.LayerNorm(dim, eps=1e-6),
                )
            )

        self.pos = PositionalEncoding(dim // 2, normalize=True)
        self.level_embed = nn.ParameterList(
            [nn.Parameter(p) for p in torch.zeros(len(strides), dim).unbind(0)]
        )

        self.encoder = encoder
        self.output_ln = nn.LayerNorm(dim, eps=1e-6)

        self.init_weights()

    def init_weights(self):
        self.apply(init_weights)

    def forward(self, batch):
        feats_all = self.backbone(batch.images)
        feats, pos, mask, shapes = self.prepare_inps(
            feats_all[self.feat_ids[0] : self.feat_ids[-1] + 1], batch.masks
        )

        enc_mask = torch.unsqueeze(mask, 1).expand(-1, mask.shape[-1], -1)
        enc_mask = enc_mask.unsqueeze(1)
        # enc_mask = None

        encoded_feats = self.encoder(feats + pos, enc_mask)

        return self.output_ln(encoded_feats), mask, shapes

    def prepare_inps(self, feats, mask):
        """
        Prepare inputs

        Args:
            feats (List[Tensor]): backbone features
                each feature has size of (B, ?, h/stride, w/stride)
            mask (Tensor[bool]): (B, H, W) mask

        Returns:
            feats (Tensor[float]): (B, ?, dim)
            pos (Tensor[float]): (B, ?, dim)
            mask (Tensor[bool]): (B, ?)
            shapes (Tensor[int64]): feature shapes (n_levels, 2)
        """
        feats_encode, feats_pos, masks, shapes = [], [], [], []
        mask = mask.unsqueeze(1).float()  # unsqueeze for convinience

        # Feature wise projection & prepare mask, positional encoding
        for input_proj, feat in zip(self.input_proj, feats):
            _, _, height, width = feat.shape

            feat_encode = input_proj(torch.permute(feat, (0, 2, 3, 1)))
            feat_encode = feat_encode.permute(0, 3, 1, 2)

            mask_resize = F.interpolate(mask, size=(height, width)).squeeze(1).bool()
            pos = self.pos(mask_resize)

            masks.append(mask_resize)  # mask_resize's shape == (B, h, w).
            feats_encode.append(feat_encode.flatten(2))
            feats_pos.append(pos.flatten(2))
            shapes.append((height, width))

        feats = torch.cat(feats_encode, 2).transpose(1, 2)
        pos = torch.cat(feats_pos, 2).transpose(1, 2)
        mask = torch.cat([m.flatten(1) for m in masks], 1)
        shapes = torch.as_tensor(shapes, dtype=torch.long, device=feats.device)

        return feats, pos, mask, shapes


class VisionTransformer(nn.Module):
    """
    VisionTransformer: Pure Vision Transformer (ViT, Swin)
    Args:
        backbone (Tuple): (Backbone Model, backbone_out_feature_dimensions)
        strides (List[int]): Backbone feature selection via strides.
            i.e. if strides == [8, 16],
            two backbone features which has stride [8, 16] is selected.
        dim (int): hidden dimension
    """

    def __init__(
        self,
        backbone: nn.Module,
        strides: List[StrictInt],
        wo_source_pos_encod: StrictBool = False,
    ):
        super().__init__()

        self.backbone, self.backbone_dims = backbone
        feat_ids = [int(math.log(stride, 2)) - 1 for stride in strides]
        self.feat_ids = feat_ids

        input_dim = self.backbone_dims[1]
        self.pos = PositionalEncoding(input_dim // 2, normalize=True)

        self.wo_source_pos_encod = wo_source_pos_encod

        # self.init_weights()

    # def init_weights(self):
    #     self.apply(init_weights)

    def forward(self, batch):
        patch_size = self.backbone.patch_embed.patch_size
        if self.wo_source_pos_encod:
            feats_all = self.backbone(batch.images)
        else:
            input_pos = self.compute_input_pos(patch_size, batch.masks)
            feats_all = self.backbone(batch.images, input_pos)
        feats, mask, shapes = self.prepare_inps(
            feats_all[self.feat_ids[0] : self.feat_ids[-1] + 1], batch.masks
        )

        return feats, mask, shapes

    def compute_input_pos(self, patch_size, mask):
        _, height, width = mask.size()
        height, width = height // patch_size[0], width // patch_size[1]
        mask = mask.unsqueeze(1).float()  # unsqueeze for convinience

        mask_resize = F.interpolate(mask, size=(height, width)).squeeze(1).bool()
        pos = self.pos(mask_resize)
        return pos

    def prepare_inps(self, feats, mask):
        """
        Prepare inputs

        Args:
            feats (List[Tensor]): backbone features
                each feature has size of (B, ?, h/stride, w/stride)
            mask (Tensor[bool]): (B, H, W) mask

        Returns:
            feats (Tensor[float]): (B, ?, dim)
            mask (Tensor[bool]): (B, ?)
            shapes (Tensor[int64]): feature shapes (n_levels, 2)
        """
        feats_encode, masks, shapes = [], [], []
        mask = mask.unsqueeze(1).float()  # unsqueeze for convinience

        # Feature wise projection & prepare mask, positional encoding
        for feat in feats:
            _, _, height, width = feat.shape

            mask_resize = F.interpolate(mask, size=(height, width)).squeeze(1).bool()

            masks.append(mask_resize)  # mask_resize's shape == (B, h, w).
            feats_encode.append(feat.flatten(2))
            shapes.append((height, width))

        feats = torch.cat(feats_encode, 2).transpose(1, 2)
        mask = torch.cat([m.flatten(1) for m in masks], 1)
        shapes = torch.as_tensor(shapes, dtype=torch.long, device=feats.device)

        return feats, mask, shapes


class Units(nn.Module):
    """
    Units Model
    Args:
        dim_enc (int): hidden dimension
        dim_dec (int): hidden dimension
        encoder (nn.Module): encoder (backbone + transformer, i.e HybridTransformer, ViT, Swin)
        decoder (nn.Module): decoder (auto-regressive decoder)
    """

    def __init__(
        self,
        dim_enc: StrictInt,
        dim_dec: StrictInt,
        encoder: nn.Module,
        decoder: nn.Module,
        wo_source_pos_encod: StrictBool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        # Connector between encoders and decoders
        self.connector = nn.Sequential(
            nn.Linear(dim_enc, dim_dec), nn.LayerNorm(dim_dec, eps=1e-6)
        )
        self.proj_mlp = MLP(1, dim_dec, 4, drop_path=0.1, drop_units=0.1)

        self.pos = PositionalEncoding(dim_dec // 2, normalize=True)
        self.wo_source_pos_encod = wo_source_pos_encod

        self.init_weights()

    def init_weights(self):
        # Encoder should not be initialized since it utilizes a pre-trained model
        # self.apply(init_weights)
        self.decoder.apply(init_weights)
        self.connector.apply(init_weights)
        self.proj_mlp.apply(init_weights)

    def forward(self, batch, detect_type=None):
        encoded_feats, mask, shapes = self.encoder(batch)

        encoded_feats = self.connector(encoded_feats)
        if self.wo_source_pos_encod:
            encoded_feats = self.proj_mlp(encoded_feats)
        else:
            pos = self.compute_pos_embed(shapes, batch.masks)
            encoded_feats = self.proj_mlp(encoded_feats + pos)

        outputs = self.decoder(
            batch,
            encoded_feats,
            mask,
            detect_type,
        )

        return outputs

    def compute_pos_embed(self, shapes, mask):
        """
        compute pos embed for input 2D features

        Args:
            shapes (Tensor[int64]): feature shapes (n_levels, 2)
            mask (Tensor[bool]): (B, H, W) mask

        Returns:
            pos (Tensor[float]): (B, ?, dim)
        """
        feats_pos = []
        mask = mask.unsqueeze(1).float()  # unsqueeze for convinience

        for shape in shapes:
            height, width = shape.tolist()

            mask_resize = F.interpolate(mask, size=(height, width)).squeeze(1).bool()
            pos = self.pos(mask_resize)

            feats_pos.append(pos.flatten(2))

        pos = torch.cat(feats_pos, 2).transpose(1, 2)

        return pos
