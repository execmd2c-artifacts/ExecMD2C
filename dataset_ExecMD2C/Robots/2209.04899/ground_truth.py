from typing import Dict, List

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from scipy.spatial.transform import Rotation as R

from openpoints.models.backbone.pointnext import (
    PointNextEncoder, FeaturePropogation
)


class BaseModel(nn.Module):
    @property
    def num_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            nweights += np.prod(v.size())
            nparams += 1
        return nweights, nparams

    @property
    def num_trainable_parameters(self):
        nweights, nparams = 0, 0
        for k, v in self.named_parameters():
            if v.requires_grad:
                nweights += np.prod(v.size())
                nparams += 1
        return nweights, nparams

    def prepare_batch(self, batch):
        device = next(self.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch


def dense_layer(in_channels, out_channels, apply_activation=True):
    layer: List[nn.Module] = [nn.Linear(in_channels, out_channels)]
    if apply_activation:
        layer += [nn.LeakyReLU(0.02)]
    return layer


def normalise_quat(x):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


class ActionLoss(object):
    def __init__(self, use_discrete_rot: bool = False, rot_resolution: int = 5):
        self.use_discrete_rot = use_discrete_rot
        if self.use_discrete_rot:
            self.rot_resolution = rot_resolution
            self.rot_classes = 360 // rot_resolution

    def decompose_actions(self, actions, onehot_rot=False):
        pos = actions[..., :3]
        if not self.use_discrete_rot:
            rot = actions[..., 3:7]
            open = actions[..., 7]
        else:
            if onehot_rot:
                rot = actions[..., 3: 6].long()
            else:
                rot = [
                    actions[..., 3: 3 + self.rot_classes],
                    actions[..., 3 + self.rot_classes: 3 + 2*self.rot_classes],
                    actions[..., 3 + 2*self.rot_classes: 3 + 3*self.rot_classes],
                ]
            open = actions[..., -1]
        return pos, rot, open

    def compute_loss(
            self, preds, targets, masks=None,
            heatmap_loss=False, distance_weight=1, heatmap_loss_weight=1,
            pred_heatmap_logits=None, pred_offset=None, pcd_xyzs=None,
            use_heatmap_max=False, use_pos_loss=True
        ) -> Dict[str, torch.Tensor]:
        pred_pos, pred_rot, pred_open = self.decompose_actions(preds)
        tgt_pos, tgt_rot, tgt_open = self.decompose_actions(targets, onehot_rot=True)

        losses = {}
        losses['pos'] = F.mse_loss(pred_pos, tgt_pos)

        if self.use_discrete_rot:
            losses['rot'] = (F.cross_entropy(pred_rot[0], tgt_rot[:, 0]) +
                            F.cross_entropy(pred_rot[1], tgt_rot[:, 1]) +
                            F.cross_entropy(pred_rot[2], tgt_rot[:, 2])) / 3
        else:
            # Automatically matching the closest quaternions (symmetrical solution).
            tgt_rot_ = -tgt_rot.clone()
            rot_loss = F.mse_loss(pred_rot, tgt_rot, reduction='none').mean(-1)
            rot_loss_ = F.mse_loss(pred_rot, tgt_rot_, reduction='none').mean(-1)
            select_mask = (rot_loss < rot_loss_).float()
            losses['rot'] = (select_mask * rot_loss + (1 - select_mask) * rot_loss_).mean()

        losses['open'] = F.binary_cross_entropy_with_logits(pred_open, tgt_open)

        if use_pos_loss:
            losses['total'] = losses['pos'] + losses['rot'] + losses['open']
        else:
            losses['total'] = losses['rot'] + losses['open']

        if heatmap_loss:
            # (batch, npoints, 3)
            tgt_offset = targets[:, :3].unsqueeze(1) - pcd_xyzs
            dists = torch.norm(tgt_offset, dim=-1)
            if use_heatmap_max:
                tgt_heatmap_index = torch.min(dists, 1)[1]  # (b, )

                losses['xt_heatmap'] = F.cross_entropy(
                    pred_heatmap_logits, tgt_heatmap_index
                )
                losses['total'] += losses['xt_heatmap'] * heatmap_loss_weight

                losses['xt_offset'] = F.mse_loss(
                    pred_offset.gather(
                        2, einops.repeat(tgt_heatmap_index, 'b -> b 3').unsqueeze(2)
                    ),
                    tgt_offset.gather(
                        1, einops.repeat(tgt_heatmap_index, 'b -> b 3').unsqueeze(1)
                    )
                )
                losses['total'] += losses['xt_offset']

            else:
                inv_dists = 1 / (1e-12 + dists)**distance_weight

                tgt_heatmap = torch.softmax(inv_dists, dim=1)
                tgt_log_heatmap = torch.log_softmax(inv_dists, dim=1)
                losses['tgt_heatmap_max'] = torch.mean(tgt_heatmap.max(1)[0])

                losses['xt_heatmap'] = F.kl_div(
                    torch.log_softmax(pred_heatmap_logits, dim=-1), tgt_log_heatmap,
                    reduction='batchmean', log_target=True
                )
                losses['total'] += losses['xt_heatmap'] * heatmap_loss_weight

                losses['xt_offset'] = torch.sum(F.mse_loss(
                    pred_offset.permute(0, 2, 1), tgt_offset,
                    reduction='none'
                ) * tgt_heatmap.unsqueeze(2)) / tgt_offset.size(0) / 3

                losses['total'] += losses['xt_offset']

        return losses


class PositionalEncoding(nn.Module):
    '''
        Transformer-style positional encoding with wavelets
    '''

    def __init__(self, dim_embed, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, dim_embed)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim_embed, 2, dtype=torch.float) *
                -(math.log(10000.0) / dim_embed)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        self.pe = pe # size=(max_len, dim_embed)
        self.dim_embed = dim_embed

    def forward(self, step_ids):
        if step_ids.device != self.pe.device:
            self.pe = self.pe.to(step_ids.device)
        return self.pe[step_ids]


class PointNextDecoder(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        fp_channels = encoder_channel_list[:-1]     # feature propogation

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i]
            )
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f, txt_tokens=None, txt_padding_masks=None, return_all_layers=False):
        if return_all_layers:
            out_per_layer = []

        for i in range(-1, -len(self.decoder) - 1, -1):
            x = self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])
            f[i - 1] = self.decoder[i][1:]([p[i], x])[1]
            if return_all_layers:
                out_per_layer.append(f[i - 1])

        if return_all_layers:
            return out_per_layer

        out = f[-len(self.decoder) - 1]
        return out


class ActionHead(nn.Module):
    def __init__(
            self, dec_channels, heatmap_temp=1, dropout=0, use_max_action=False,
            use_discrete_rot: bool = False, rot_resolution: int = 5,
        ) -> None:
        super().__init__()
        self.use_discrete_rot = use_discrete_rot
        self.rot_resolution = rot_resolution

        if self.use_discrete_rot:
            self.rot_decoder = nn.Sequential(
                *dense_layer(dec_channels[0], dec_channels[0] // 2),
                nn.Dropout(dropout),
                *dense_layer(dec_channels[0] // 2, (360 // rot_resolution) * 3 + 1, apply_activation=False),
            )
        else:
            self.quat_decoder = nn.Sequential(
                *dense_layer(dec_channels[0], dec_channels[0] // 2),
                nn.Dropout(dropout),
                *dense_layer(dec_channels[0] // 2, 4 + 1, apply_activation=False),
            )

        self.maps_to_coord = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(dec_channels[-1], 1 + 3, 1)
        )
        self.heatmap_temp = heatmap_temp
        self.use_max_action = use_max_action

    def forward(self, dec_fts, pcds, pc_centers, pc_radii):
        '''
        - dec_fts: [(batch, dec_channels[0], npoints), (batch, dec_channels[-1], npoints)]
        - pcds: (batch, 3, npoints)
        '''
        # predict the translation of the gripper
        xt_fts = self.maps_to_coord(dec_fts[-1])
        xt_heatmap = torch.softmax(xt_fts[:, :1] / self.heatmap_temp, dim=-1)
        xt_offset = xt_fts[:, 1:]
        if self.use_max_action:
            xt = pcds + xt_offset   # (b, 3, npoints)
            xt = xt.gather(
                2, einops.repeat(torch.max(xt_heatmap, dim=2)[1], 'b 1 -> b 3').unsqueeze(2)
            ).squeeze(2)
        else:
            xt = einops.reduce((pcds + xt_offset) * xt_heatmap, 'b c n -> b c', 'sum')
        xt = xt * pc_radii + pc_centers

        # predict the (rotation, openness) of the gripper
        xg_fts, _ = torch.max(dec_fts[0], -1)

        if self.use_discrete_rot:
            xg = self.rot_decoder(xg_fts)
            xr = xg[..., :-1]
        else:
            xg = self.quat_decoder(xg_fts)
            xr = normalise_quat(xg[..., :4])

        xo = xg[..., -1:]

        actions = torch.cat([xt, xr, xo], dim=-1)

        return {
            'actions': actions,
            'xt_offset': xt_offset * pc_radii.unsqueeze(2),
            'xt_heatmap': xt_heatmap.squeeze(1),
            'xt_heatmap_logits': xt_fts[:, 0] / self.heatmap_temp,
        }


class ActionEmbedding(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()

        self.open_embedding = nn.Embedding(2, hidden_size)
        self.pos_embedding = nn.Linear(3, hidden_size)
        self.rot_embedding = nn.Linear(6, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, actions):
        '''
        actions: (batch_size, 8)
        '''
        pos_embeds = self.pos_embedding(actions[..., :3])
        open_embeds = self.open_embedding(actions[..., -1].long())

        rot_euler_angles = R.from_quat(actions[..., 3:7].data.cpu()).as_euler('xyz')
        rot_euler_angles = torch.from_numpy(rot_euler_angles).float().to(actions.device)
        rot_inputs = torch.cat(
            [torch.sin(rot_euler_angles), torch.cos(rot_euler_angles)], -1
        )
        rot_embeds = self.rot_embedding(rot_inputs)

        act_embeds = self.layer_norm(
            pos_embeds + rot_embeds + open_embeds
        )
        return act_embeds


class PointCloudUNet(BaseModel):
    def __init__(
        self, pcd_encoder_cfg, pcd_decoder_cfg,
        num_tasks: int = None, max_steps: int = 20,
        use_instr_embed: str = 'none', instr_embed_size: int = None,
        txt_attn_type: str = 'none', num_trans_layers: int = 1,
        trans_hidden_size: int = 512,
        dropout=0.2, heatmap_temp=1, use_prev_action=False,
        cat_global_in_head=False, **kwargs
    ):
        super().__init__()

        self.pcd_encoder_cfg = pcd_encoder_cfg
        self.pcd_decoder_cfg = pcd_decoder_cfg
        self.num_tasks = num_tasks
        self.max_steps = max_steps
        self.use_instr_embed = use_instr_embed
        self.instr_embed_size = instr_embed_size
        self.txt_attn_type = txt_attn_type
        self.num_trans_layers = num_trans_layers
        self.use_prev_action = use_prev_action
        self.cat_global_in_head = cat_global_in_head
        self.heatmap_temp = heatmap_temp
        self.use_discrete_rot = kwargs.get('use_discrete_rot', False)
        self.rot_resolution = kwargs.get('rot_resolution', 5)
        self.kwargs = kwargs

        self.pcd_encoder = PointNextEncoder(**pcd_encoder_cfg)
        enc_channel_list = self.pcd_encoder.channel_list
        self.hidden_size = trans_hidden_size

        self.pcd_decoder = PointNextDecoder(
            enc_channel_list[:-1] + [enc_channel_list[-1] + self.hidden_size], pcd_decoder_cfg.layers,
        )

        if self.kwargs.get('learnable_step_embedding', True):
            self.step_embedding = nn.Embedding(self.max_steps, self.hidden_size)
        else:
            self.step_embedding = PositionalEncoding(self.hidden_size, max_len=self.max_steps)

        if self.use_prev_action:
            self.prev_action_embedding = ActionEmbedding(self.hidden_size)

        if self.use_instr_embed == 'none':
            assert self.num_tasks is not None
            self.task_embedding = nn.Embedding(self.num_tasks, self.hidden_size)
        else:
            assert self.instr_embed_size is not None
            self.task_embedding = nn.Linear(self.instr_embed_size, self.hidden_size)

        self.point_pos_embedding = nn.Linear(3, self.hidden_size)

        if self.txt_attn_type == 'cross':
            if enc_channel_list[-1] != self.hidden_size:
                self.pcd_to_trans_fc = nn.Conv1d(
                    in_channels=enc_channel_list[-1],
                    out_channels=self.hidden_size,
                    kernel_size=1, stride=1
                )
            else:
                self.pcd_to_trans_fc = None
            trans_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size*4,
                dropout=0.1, activation='gelu',
                layer_norm_eps=1e-12, norm_first=False,
                batch_first=True,
            )
            self.cross_attention = nn.TransformerDecoder(
                trans_layer, num_layers=self.num_trans_layers
            )

        dec_ft_size = enc_channel_list[0]
        if self.cat_global_in_head:
            dec_ft_size += self.hidden_size
        self.head = ActionHead(
            [enc_channel_list[-1] + self.hidden_size, dec_ft_size],
            heatmap_temp=heatmap_temp, dropout=dropout,
            use_max_action=kwargs.get('use_max_action', False),
            use_discrete_rot=self.use_discrete_rot,
            rot_resolution=self.rot_resolution,
        )

        self.loss_fn = ActionLoss(
            use_discrete_rot=self.use_discrete_rot,
            rot_resolution=self.rot_resolution
        )

    def forward(self, batch, compute_loss=False):
        batch = self.prepare_batch(batch)

        # encode point cloud
        pcd_fts = batch['fts']  # (batch, dim, npoints)
        pcd_poses = pcd_fts[:, :3]

        pos_list, ft_list = self.pcd_encoder(
            pcd_poses.permute(0, 2, 1).contiguous(), pcd_fts
        )
        ctx_embeds = ft_list[-1]
        if self.pcd_to_trans_fc is not None:
            ctx_embeds = self.pcd_to_trans_fc(ctx_embeds)

        step_ids = batch['step_ids']
        step_embeds = self.step_embedding(step_ids)
        ctx_embeds = ctx_embeds + step_embeds.unsqueeze(2)
        if self.use_prev_action:
            ctx_embeds = ctx_embeds + self.prev_action_embedding(batch['prev_actions']).unsqueeze(2)
        ctx_embeds = ctx_embeds + self.point_pos_embedding(pos_list[-1]).permute(0, 2, 1)

        # conditioned on the task
        taskvar_ids = batch['taskvar_ids']
        instr_embeds = batch.get('instr_embeds', None)
        txt_masks = batch.get('txt_masks', None)

        if self.use_instr_embed == 'none':
            task_embeds = self.task_embedding(taskvar_ids).unsqueeze(1)  # (batch, 1, dim)
        else:
            task_embeds = self.task_embedding(instr_embeds) # (batch, 1/len, dim)

        if self.txt_attn_type == 'none':
            assert task_embeds.size(1) == 1
            ctx_embeds = ctx_embeds + task_embeds.permute(0, 2, 1)
        elif self.txt_attn_type == 'cross':
            assert txt_masks is not None
            ctx_embeds = self.cross_attention(
                ctx_embeds.permute(0, 2, 1), task_embeds,
                memory_key_padding_mask=txt_masks.logical_not(),
            )
            ctx_embeds = ctx_embeds.permute(0, 2, 1)
        else:
            raise NotImplementedError(f'unsupported txt_attn_type {self.txt_attn_type}')

        ft_list[-1] = torch.cat([ft_list[-1], ctx_embeds], dim=1)

        # decoding features
        dec_fts = self.pcd_decoder(pos_list, ft_list)

        if self.cat_global_in_head:
            global_ctx_embeds, _ = torch.max(ctx_embeds, 2)
            global_ctx_embeds = einops.repeat(global_ctx_embeds, 'b c -> b c n', n=dec_fts.size(2))
            dec_fts = torch.cat([dec_fts, global_ctx_embeds], dim=1)
        outs = self.head(
            (ft_list[-1], dec_fts), pcd_poses,
            batch['pc_centers'], batch['pc_radii']
        )
        actions = outs['actions']

        if compute_loss:
            heatmap_loss = self.kwargs.get('heatmap_loss', False)
            heatmap_loss_weight = self.kwargs.get('heatmap_loss_weight', 1)
            distance_weight = self.kwargs.get('heatmap_distance_weight', 1)
            if heatmap_loss:
                pcd_xyzs = pcd_poses.permute(0, 2, 1) * batch['pc_radii'].unsqueeze(1) + batch['pc_centers'].unsqueeze(1) # (b, npoints, 3)
            else:
                pcd_xyzs = None
            losses = self.loss_fn.compute_loss(
                actions, batch['actions'], heatmap_loss=heatmap_loss,
                pred_heatmap_logits=outs['xt_heatmap_logits'],
                pred_offset=outs['xt_offset'],
                pcd_xyzs=pcd_xyzs, distance_weight=distance_weight,
                heatmap_loss_weight=heatmap_loss_weight,
                use_heatmap_max=self.kwargs.get('use_heatmap_max', False),
                use_pos_loss=self.kwargs.get('use_pos_loss', True)
            )

            return losses, actions

        return actions


if __name__ == "__main__":
    torch.manual_seed(7)
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        global passed, failed
        if condition:
            passed += 1
            print(f"PASS: {name}")
        else:
            failed += 1
            print(f"FAIL: {name} {detail}")

    def skip_checks(prefix, reason, count):
        global failed
        failed += count
        for idx in range(count):
            print(f"FAIL: {prefix} #{idx + 1} skipped: {reason}")

    try:
        layers = dense_layer(3, 5)
        check("dense_layer creates linear then activation", len(layers) == 2 and isinstance(layers[0], nn.Linear) and isinstance(layers[1], nn.LeakyReLU))
        no_activation = dense_layer(3, 5, apply_activation=False)
        check("dense_layer can omit activation", len(no_activation) == 1 and isinstance(no_activation[0], nn.Linear))

        quat = torch.tensor([[3.0, 4.0, 0.0, 0.0], [1.0, -2.0, 2.0, -4.0]])
        normed = normalise_quat(quat)
        check("normalise_quat returns unit quaternions", torch.allclose(normed.norm(dim=-1), torch.ones(2), atol=1e-6))
        check("normalise_quat preserves direction ratios", torch.allclose(normed[0], torch.tensor([0.6, 0.8, 0.0, 0.0]), atol=1e-6))
    except Exception as exc:
        skip_checks("basic utility checks", repr(exc), 4)

    try:
        pos = PositionalEncoding(6, max_len=10)
        step_ids = torch.tensor([0, 1, 4, 7])
        encoded = pos(step_ids)
        check("PositionalEncoding preserves requested shape", tuple(encoded.shape) == (4, 6))
        check("PositionalEncoding first row has sin zero cos one", torch.allclose(encoded[0], torch.tensor([0., 1., 0., 1., 0., 1.]), atol=1e-6))
        check("PositionalEncoding indexes different steps differently", not torch.allclose(encoded[1], encoded[2]))
    except Exception as exc:
        skip_checks("positional encoding checks", repr(exc), 3)

    try:
        continuous_loss = ActionLoss(use_discrete_rot=False)
        preds = torch.tensor([
            [0.1, 0.2, 0.3, 0.5, -0.5, 0.5, -0.5, 3.0],
            [1.0, 2.0, 3.0, -0.5, 0.5, -0.5, 0.5, -3.0],
        ])
        targets = torch.tensor([
            [0.1, 0.2, 0.3, -0.5, 0.5, -0.5, 0.5, 1.0],
            [1.0, 2.0, 3.0, 0.5, -0.5, 0.5, -0.5, 0.0],
        ])
        losses = continuous_loss.compute_loss(preds, targets)
        check("ActionLoss continuous exposes all base losses", {"pos", "rot", "open", "total"}.issubset(losses.keys()))
        check("ActionLoss continuous handles quaternion sign symmetry", losses["rot"].item() < 1e-7)

        logits = torch.zeros(2, 3 + 12 + 1)
        logits[0, 3 + 1] = 8.0
        logits[0, 3 + 4 + 2] = 8.0
        logits[0, 3 + 8 + 3] = 8.0
        logits[1, 3 + 0] = 8.0
        logits[1, 3 + 4 + 1] = 8.0
        logits[1, 3 + 8 + 2] = 8.0
        discrete_targets = torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
        ])
        discrete_loss = ActionLoss(use_discrete_rot=True, rot_resolution=90)
        dlosses = discrete_loss.compute_loss(logits, discrete_targets)
        check("ActionLoss discrete splits three rotation axes", dlosses["rot"].item() < 0.01)
        check("ActionLoss can drop position term from total", torch.allclose(continuous_loss.compute_loss(preds, targets, use_pos_loss=False)["total"], losses["rot"] + losses["open"]))

        heatmap_losses = continuous_loss.compute_loss(
            preds,
            targets,
            heatmap_loss=True,
            pred_heatmap_logits=torch.tensor([[0.0, 3.0, 1.0], [2.0, 0.0, 1.0]]),
            pred_offset=torch.zeros(2, 3, 3),
            pcd_xyzs=torch.tensor([
                [[0.3, 0.2, 0.3], [0.1, 0.2, 0.3], [3.0, 3.0, 3.0]],
                [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 3.0, 3.0]],
            ]),
            use_heatmap_max=True,
        )
        check("ActionLoss heatmap-max adds heatmap terms", {"xt_heatmap", "xt_offset"}.issubset(heatmap_losses.keys()))
    except Exception as exc:
        skip_checks("action loss checks", repr(exc), 6)

    try:
        class ConstantDecoder(nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                return torch.tensor([[2.0, 0.0, 0.0, 0.0, -0.25]], device=x.device).repeat(batch, 1)

        head = ActionHead([8, 4], heatmap_temp=1.0, dropout=0.0, use_max_action=False)
        head.maps_to_coord = nn.Identity()
        head.quat_decoder = ConstantDecoder()
        dec_global = torch.randn(1, 8, 3)
        coord_logits_offsets = torch.tensor([[[0.0, 1.0, 2.0],
                                             [0.1, 0.1, 0.1],
                                             [0.0, 0.2, 0.4],
                                             [0.3, 0.3, 0.3]]])
        pcds = torch.tensor([[[1.0, 2.0, 3.0],
                              [0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0]]])
        center = torch.tensor([[10.0, 20.0, 30.0]])
        radius = torch.tensor([[2.0, 3.0, 4.0]])
        outs = head((dec_global, coord_logits_offsets), pcds, center, radius)
        weights = torch.softmax(coord_logits_offsets[:, :1], dim=-1)
        expected_xt = ((pcds + coord_logits_offsets[:, 1:]) * weights).sum(dim=-1) * radius + center
        check("ActionHead weighted translation matches heatmap expectation", torch.allclose(outs["actions"][:, :3], expected_xt, atol=1e-6))
        check("ActionHead normalizes continuous quaternion output", torch.allclose(outs["actions"][:, 3:7].norm(dim=-1), torch.ones(1), atol=1e-6))
        check("ActionHead returns heatmap logits without channel dimension", tuple(outs["xt_heatmap_logits"].shape) == (1, 3))
        check("ActionHead rescales offsets by point-cloud radius", torch.allclose(outs["xt_offset"], coord_logits_offsets[:, 1:] * radius.unsqueeze(2)))
    except Exception as exc:
        skip_checks("action head checks", repr(exc), 4)

    try:
        act_embed = ActionEmbedding(hidden_size=8)
        actions = torch.tensor([
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.5, -0.2, 0.0, 0.0, 0.70710677, 0.70710677, 1.0],
        ])
        embeds = act_embed(actions)
        check("ActionEmbedding returns hidden-size vectors", tuple(embeds.shape) == (2, 8))
        check("ActionEmbedding output is finite", torch.isfinite(embeds).all().item())
    except Exception as exc:
        skip_checks("action embedding checks", repr(exc), 2)

    try:
        original_fp = globals()["FeaturePropogation"]

        class FakeFeaturePropogation(nn.Module):
            def __init__(self, mlp):
                super().__init__()
                self.out_channels = mlp[-1]

            def forward(self, target, source):
                p_target, _ = target
                batch, npoints = p_target.shape[0], p_target.shape[1]
                return torch.ones(batch, self.out_channels, npoints, device=p_target.device)

        globals()["FeaturePropogation"] = FakeFeaturePropogation
        decoder = PointNextDecoder([4, 8, 16], decoder_layers=2)
        p_list = [torch.randn(2, 5, 3), torch.randn(2, 3, 3), torch.randn(2, 2, 3)]
        f_list = [torch.randn(2, 4, 5), torch.randn(2, 8, 3), torch.randn(2, 16, 2)]
        decoded = decoder(p_list, [x.clone() for x in f_list])
        all_layers = decoder(p_list, [x.clone() for x in f_list], return_all_layers=True)
        check("PointNextDecoder restores highest point resolution", tuple(decoded.shape) == (2, 4, 5))
        check("PointNextDecoder can return all decoder layers", len(all_layers) == 2 and tuple(all_layers[-1].shape) == (2, 4, 5))
        globals()["FeaturePropogation"] = original_fp
    except Exception as exc:
        globals()["FeaturePropogation"] = original_fp if "original_fp" in locals() else globals().get("FeaturePropogation")
        skip_checks("decoder checks", repr(exc), 2)

    try:
        original_encoder = globals()["PointNextEncoder"]
        original_fp = globals()["FeaturePropogation"]

        class FakeEncoder(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.channel_list = [4, 8]

            def forward(self, positions, features):
                batch, npoints = positions.shape[0], positions.shape[1]
                coarse_n = max(1, npoints // 2)
                pos0 = positions
                pos1 = positions[:, :coarse_n, :]
                f0 = features[:, :4, :]
                f1 = torch.ones(batch, 8, coarse_n, device=features.device)
                return [pos0, pos1], [f0, f1]

        class FakeFeaturePropogation(nn.Module):
            def __init__(self, mlp):
                super().__init__()
                self.out_channels = mlp[-1]

            def forward(self, target, source):
                p_target, f_target = target
                source_mean = source[1].mean(dim=(1, 2), keepdim=True)
                return f_target[:, :self.out_channels, :] + source_mean

        globals()["PointNextEncoder"] = FakeEncoder
        globals()["FeaturePropogation"] = FakeFeaturePropogation

        class Cfg:
            layers = 1

        model = PointCloudUNet(
            pcd_encoder_cfg={"in_channels": 10},
            pcd_decoder_cfg=Cfg(),
            num_tasks=1,
            max_steps=5,
            use_instr_embed="all",
            instr_embed_size=6,
            txt_attn_type="cross",
            num_trans_layers=1,
            trans_hidden_size=8,
            dropout=0.0,
            heatmap_temp=1.0,
            use_prev_action=False,
            learnable_step_embedding=False,
        )
        batch = {
            "fts": torch.randn(2, 10, 6),
            "step_ids": torch.tensor([0, 3]),
            "taskvar_ids": torch.zeros(2, dtype=torch.long),
            "instr_embeds": torch.randn(2, 3, 6),
            "txt_masks": torch.ones(2, 3, dtype=torch.bool),
            "pc_centers": torch.randn(2, 3),
            "pc_radii": torch.ones(2, 3),
            "actions": torch.tensor([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [0.5, -0.2, 0.1, 0.0, 0.0, 0.70710677, 0.70710677, 0.0],
            ]),
        }
        model.eval()
        with torch.no_grad():
            actions = model(batch)
            losses, loss_actions = model(batch, compute_loss=True)
        check("PointCloudUNet forward returns 8D continuous actions", tuple(actions.shape) == (2, 8))
        check("PointCloudUNet compute_loss returns losses and actions", "total" in losses and tuple(loss_actions.shape) == (2, 8))
        check("PointCloudUNet normalizes predicted quaternions", torch.allclose(actions[:, 3:7].norm(dim=-1), torch.ones(2), atol=1e-5))

        globals()["PointNextEncoder"] = original_encoder
        globals()["FeaturePropogation"] = original_fp
    except Exception as exc:
        globals()["PointNextEncoder"] = original_encoder if "original_encoder" in locals() else globals().get("PointNextEncoder")
        globals()["FeaturePropogation"] = original_fp if "original_fp" in locals() else globals().get("FeaturePropogation")
        skip_checks("PointCloudUNet integration checks", repr(exc), 3)

    print(f"RESULT: {passed} passed, {failed} failed")
    if failed != 0:
        raise SystemExit(1)
