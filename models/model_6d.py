
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from functools import partial
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from timm.models.vision_transformer import PatchEmbed, Block
from util.model_component import Transpose, ResConv1DBlock, SDivide
from util.misc import rot6d_to_rotmat


prev_list = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]



class AdapterModule(nn.Module):
    def __init__(self, in_feature, bottleneck):
        super().__init__()

        self.proj_down = nn.Linear(in_features=in_feature, out_features=bottleneck)
        self.proj_up = nn.Linear(in_features=bottleneck, out_features=in_feature)

    def forward(self, x):
        input = x.clone()

        x = self.proj_down(x)
        x = F.relu(x)
        return self.proj_up(x) + input # Skip Connection


class angle_loss(nn.Module):
    def __init__(self):
        super(angle_loss, self).__init__()

    def forward(self, input, target, bs):
        y_pred1 = input[..., 0, :]
        y_pred2 = input[..., 1, :]
        y_pred3 = input[..., 2, :]
        y_pred1 = y_pred1.view(-1,  1, 3)
        y_pred2 = y_pred2.view(-1, 1, 3)
        y_pred3 = y_pred3.view(-1, 1, 3)
        y_pred1 = y_pred1.repeat([1, 3, 1])
        y_pred2 = y_pred2.repeat([1,  3, 1])
        y_pred3 = y_pred3.repeat([1, 3, 1])
        target = target.view(-1, 3, 3)
        z_pred1 = target * y_pred1
        z_pred2 = target * y_pred2
        z_pred3 = target * y_pred3
        z_pred1 = torch.sum(z_pred1, axis=-1)
        z_pred2 = torch.sum(z_pred2, axis=-1)
        z_pred3 = torch.sum(z_pred3, axis=-1)
        z_pred1 = z_pred1.view(-1,  3, 1)
        z_pred2 = z_pred2.view(-1,  3, 1)
        z_pred3 = z_pred3.view(-1,  3, 1)
        z_pred = torch.cat([z_pred1, z_pred2, z_pred3], axis=2)
        z_pred_trace = z_pred[:,  0, 0] + z_pred[:,  1, 1] + z_pred[:,  2, 2]
        z_pred_trace = (z_pred_trace - 1.)/2.0000000000

        z_pred_trace = torch.clamp(z_pred_trace, -0.99999, 0.99999)
        z_pred_trace = torch.acos(z_pred_trace)
        z_pred_trace = z_pred_trace * 180./3.141592653
    

        error = torch.mean(z_pred_trace)
        return error
    




class AdapterNet(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_num_heads=16, adapter_depth=8,
                 mlp_ratio=4.,rotate_chans = 6, norm_layer=partial(nn.LayerNorm, eps=1e-6), joint_num=24, marker_num=56):
        super().__init__()

        self.joint_num = joint_num
        self.marker_num = marker_num
        self.rotate_predictor = nn.Sequential(  # per body part
            Transpose(-2, -1),
            ResConv1DBlock(3, embed_dim),
            nn.ReLU(), )
        

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.adpat_linear = nn.Linear(self.marker_num, self.joint_num, bias=True)
        self.rotate_pred = nn.Linear(embed_dim, rotate_chans, bias=True)
        self.rotate_loss = angle_loss()
        self.edges = []
        for i in range(1, joint_num):
            self.edges.append((prev_list[i], i))
        self.topology = [-1] * (len(self.edges) + 1)
        for i, edge in enumerate(self.edges):
            self.topology[edge[1]] = edge[0]

        self.apply(self._init_weights)

        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def compute_offsets(self, points):
        """
        given a batch of seq of points compute the center of the points at first time frame
        this is basically th bos offset
        Args:
            points: Nxnum_pointsx3

        Returns:
            Nx1x3

        """
        bs, num_markers, _ = points.shape

        nonzero_mask = ((points == 0.0).sum(-1) != 3)
        offsets = []
        for i in range(bs):
            if nonzero_mask[i].sum() == 0:
                offsets.append(points.new(torch.zeros([1,3])))
                continue
            offsets.append(torch.median(points[i, nonzero_mask[i]], dim=0, keepdim=True).values)
        return torch.cat(offsets, dim=0).view(bs, 1, 3)

    def forward(self, x, gt, is_valid=False):
        N = x.shape[0]
        x = x[:, :56]
        pts_offset = self.compute_offsets(x)
        x = x - pts_offset
        x = self.rotate_predictor(x)
        x = torch.transpose(x, 1, 2)
        #assert x.shape[1] == self.marker_num + self.joint_num
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.adpat_linear(x.transpose(1, 2)).transpose(1, 2)
        x_6d = self.rotate_pred(x)
        bs = gt.shape[0]

        R_pred = rot6d_to_rotmat(x_6d.reshape(-1, 6))
        rotate_loss = self.rotate_loss(R_pred, gt.reshape(-1, 3, 3), bs)
        return R_pred, rotate_loss
