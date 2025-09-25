
from functools import partial
from util.model_component import Transpose, ResConv1DBlock, SDivide

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=None, patch_size=1, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, joint_num=24, marker_num=56):
        super().__init__()

        self.joint_num = joint_num
        self.marker_num = marker_num
        #self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        #num_patches = self.patch_embed.num_patches
        num_patches = marker_num
        self.score_predictor_b1 = nn.Sequential(  # per body part
            Transpose(-2, -1),
            ResConv1DBlock(3, embed_dim),
            nn.ReLU(), )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.pts_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.joint_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.marker_num + self.joint_num + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        '''
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        '''
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], self.marker_num, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_pos_embed.shape[-1], self.marker_num + self.joint_num, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))


        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        #w = self.patch_embed.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        torch.nn.init.normal_(self.pts_token, std=.02)
        torch.nn.init.normal_(self.joint_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
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
                offsets.append(points.new(torch.zeros([1,3]).to('cuda')))
                continue
            offsets.append(torch.median(points[i, nonzero_mask[i]], dim=0, keepdim=True).values)
        return torch.cat(offsets, dim=0).view(bs, 1, 3)
    



    def forward_encoder(self, x):
        # embed patches
        mask = torch.all(x == 0, axis=-1)  # shape (N, L)
        pts_offset = self.compute_offsets(x)
        x = x - pts_offset

        #x = self.patch_embed(x)
        x = self.score_predictor_b1(x)
        x = torch.transpose(x, 1, 2)


        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        #return x, mask, ids_restore, pts_offset
        return x, mask, pts_offset

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        #x_ = torch.gather(x[:, 1:, :], dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # append mask tokens to sequence
        joint_tokens = self.joint_token.repeat(x.shape[0], self.joint_num, 1)
        x = torch.cat([x.clone(), joint_tokens], dim=1)  # no cls token
        #x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        decode_f = x
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return decode_f, x

    def forward_loss(self, gt_marker, gt_joint, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """

        marker_loss = (torch.linalg.norm(pred[:,:self.marker_num] - gt_marker, dim=-1)) ** 2
        #marker_loss = m_loss.mean(dim=-1)  # [N, L], mean loss per patch

        remarker_loss = (marker_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        orimarker_loss = (marker_loss * (~mask)).sum() / (~mask).sum()

        out_remarker = ((torch.linalg.norm(pred[:,:self.marker_num] - gt_marker, dim=-1)) * mask).sum() / mask.sum()
        #print(torch.mean(torch.linalg.norm(pred[:, self.marker_num:] - gt_joint, dim = -1), axis=0))
        joint_ploss = (torch.linalg.norm(pred[:, self.marker_num:] - gt_joint, dim = -1)).mean()
        mean_mloss = torch.linalg.norm(pred[:,:self.marker_num] - gt_marker, dim=-1).mean()

        j_loss = (torch.linalg.norm(pred[:, self.marker_num:] - gt_joint, dim = -1)) ** 2
        joint_loss = j_loss.mean()

        loss = remarker_loss + orimarker_loss + 2 * joint_loss

        return out_remarker, joint_ploss, mean_mloss, loss


    def forward(self, pts, is_valid=False):
        latent, mask, pts_offset = self.forward_encoder(pts['M1'])
        decode_f, pred = self.forward_decoder(latent)
        pred = pred + pts_offset  # [N, L, p*p*3]
        if is_valid:
            return None, None, None, None, pred, None
        remarker_loss, joint_ploss, mean_marker_loss, loss = self.forward_loss(pts['M'], pts['J'], pred, mask)
        return loss, remarker_loss, joint_ploss, mean_marker_loss, pred, decode_f




def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
