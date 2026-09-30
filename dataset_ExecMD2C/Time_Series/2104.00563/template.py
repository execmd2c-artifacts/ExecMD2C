#!/usr/bin/env python3
"""AutoBots core model components for an LLM reproduction benchmark."""

# ============================================================
# ground_truth.py - AutoBots Core Model Components
# Source: Time_Series/AutoBots-master
#
# Contains ONLY model architecture components and direct objective
# helpers for AutoBot-Joint. No training loop, evaluation, dataset,
# checkpoint, CLI, or inference pipeline code.
# ============================================================

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import special
import torch.distributions as D
from torch.distributions import MultivariateNormal, Laplace


# --- [Original file: models/context_encoders.py] ---
def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MapEncoderPtsMA(nn.Module):
    '''
    This class operates on the multi-agent road lanes provided as a tensor with shape
    (B, num_agents, num_road_segs, num_pts_per_road_seg, k_attr+1)
    '''
    def __init__(self, d_k, map_attr=3, dropout=0.1):
        super(MapEncoderPtsMA, self).__init__()
        self.dropout = dropout
        self.d_k = d_k
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.map_attr = map_attr

        # Seed parameters for the map
        self.map_seeds = nn.Parameter(torch.Tensor(1, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.map_seeds)

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k*3)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k*3, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        """
        [TODO] Build road segment and road point masks for multi-agent lane sets.

        Input:
            roads: (batch, agents, segments, points, map_attr+1) - lane point features with an existence flag in the last channel.

        Output:
            road_segment_mask: (batch, agents, segments) - True where a full lane segment is missing.
            road_pts_mask: (batch*agents*segments, points) - True where individual points are missing.

"""
        pass

    def forward(self, roads, agents_emb):
        """
        [TODO] Encode multi-agent lane point sets with learned map seeds.

        Input:
            roads: (batch, agents, segments, points, map_attr+1) - lane point features and existence flags.
            agents_emb: (obs_steps, batch, agents, d_k) - socio-temporal agent context from the AutoBot encoder.

        Output:
            road_seg_emb: (segments, batch, agents, d_k) - encoded lane segment features.
            road_segment_mask: (batch, agents, segments) - missing-segment mask.

"""
        pass


# --- [Original file: models/autobot_joint.py] ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: must be (T, B, H)
        :return:
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Joint's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution and possibly predicts the yaw.
    '''
    def __init__(self, d_k=64, predict_yaw=False):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        self.predict_yaw = predict_yaw
        out_len = 5
        if predict_yaw:
            out_len = 6

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(),
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(),
            init_(nn.Linear(self.d_k, out_len))
        )
        self.min_stdev = 0.01

    def forward(self, agent_latent_state):
        """
        [TODO] Convert decoder latent states into stable bivariate trajectory parameters.

        Input:
            agent_latent_state: (future_steps, batch_modes_agents, d_k) - decoder hidden states.

        Output:
            params: (future_steps, batch_modes_agents, 5 or 6) - x/y means, positive scales, bounded correlation, and optional yaw.

"""
        pass


class AutoBotJoint(nn.Module):
    '''
    AutoBot-Joint Class.
    '''
    def __init__(self, d_k=128, _M=5, c=5, T=30, L_enc=1, dropout=0.0, k_attr=2, map_attr=3, num_heads=16, L_dec=1,
                 tx_hidden_size=384, use_map_lanes=False, num_agent_types=None, predict_yaw=False):
        super(AutoBotJoint, self).__init__()

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.k_attr = k_attr
        self.map_attr = map_attr
        self.d_k = d_k
        self._M = _M  # num agents other then the main agent.
        self.c = c
        self.T = T
        self.L_enc = L_enc
        self.dropout = dropout
        self.num_heads = num_heads
        self.L_dec = L_dec
        self.tx_hidden_size = tx_hidden_size
        self.use_map_lanes = use_map_lanes
        self.predict_yaw = predict_yaw

        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))

        # ============================== AutoBot-Joint ENCODER ==============================
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=2))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        # ============================== MAP ENCODER ==========================
        if self.use_map_lanes:
            self.map_encoder = MapEncoderPtsMA(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
            self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)

        # ============================== AGENT TYPES Encoders ==============================
        self.emb_agent_types = nn.Sequential(init_(nn.Linear(num_agent_types, self.d_k)))
        self.dec_agenttypes_encoder = nn.Sequential(
            init_(nn.Linear(2 * self.d_k, self.d_k)), nn.ReLU(),
            init_(nn.Linear(self.d_k, self.d_k))
        )

        # ============================== AutoBot-Joint DECODER ==============================
        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        self.social_attn_decoder_layers = []
        self.temporal_attn_decoder_layers = []
        for _ in range(self.L_dec):
            tx_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_decoder_layers.append(nn.TransformerDecoder(tx_decoder_layer, num_layers=2))
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.social_attn_decoder_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_decoder_layers = nn.ModuleList(self.temporal_attn_decoder_layers)
        self.social_attn_decoder_layers = nn.ModuleList(self.social_attn_decoder_layers)

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0)

        # ============================== OUTPUT MODEL ==============================
        self.output_model = OutputModel(d_k=self.d_k, predict_yaw=self.predict_yaw)

        # ============================== Mode Prob prediction (P(z|X_1:t)) ==============================
        self.P = nn.Parameter(torch.Tensor(c, 1, 1, d_k), requires_grad=True)  # Appendix C.2.
        nn.init.xavier_uniform_(self.P)

        if self.use_map_lanes:
            self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)

        self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))

        self.train()

    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask

    def process_observations(self, ego, agents):
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks = ego[:, :, -1]

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).type(torch.BoolTensor).to(agents.device)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        """
        [TODO] Apply temporal transformer attention independently for each agent track.

        Input:
            agents_emb: (obs_steps, batch, agents, d_k) - embedded observed agent states.
            agent_masks: (batch, obs_steps, agents) - True where an agent observation is absent.
            layer: transformer encoder module - temporal encoder block.

        Output: (obs_steps, batch, agents, d_k) - temporally contextualized agent embeddings.

"""
        pass

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        """
        [TODO] Apply social transformer attention across agents at each observed time.

        Input:
            agents_emb: (obs_steps, batch, agents, d_k) - temporally encoded agent states.
            agent_masks: (batch, obs_steps, agents) - True where agents are absent.
            layer: transformer encoder module - social encoder block.

        Output: (obs_steps, batch, agents, d_k) - social-contextualized embeddings.

"""
        pass

    def temporal_attn_decoder_fn(self, agents_emb, context, agent_masks, layer):
        """
        [TODO] Run causal temporal cross-attention in the multimodal decoder.

        Input:
            agents_emb: (future_steps, batch_modes, agents, d_k) - decoder query states.
            context: (obs_steps, batch_modes, agents, d_k) - encoded observation memory.
            agent_masks: (batch_modes, obs_steps, agents) - absence mask for memory states.
            layer: transformer decoder module - temporal decoder block.

        Output: (future_steps, batch_modes, agents, d_k) - temporally decoded states.

"""
        pass

    def social_attn_decoder_fn(self, agents_emb, agent_masks, layer):
        """
        [TODO] Run decoder social self-attention across agents for each future step.

        Input:
            agents_emb: (future_steps, batch_modes, agents, d_k) - temporally decoded states.
            agent_masks: (batch_modes, obs_steps, agents) - absence masks from observed history.
            layer: transformer encoder module - social decoder block.

        Output: (future_steps, batch_modes, agents, d_k) - socially decoded future states.

"""
        pass

    def forward(self, ego_in, agents_in, roads, agent_types):
        """
        [TODO] Run AutoBot-Joint end-to-end for multimodal joint multi-agent prediction.

        Input:
            ego_in: (batch, obs_steps, k_attr+1) - ego history with existence flag.
            agents_in: (batch, obs_steps, other_agents, k_attr+1) - other-agent histories with existence flags.
            roads: (batch, agents, segments, points, map_attr+1) or placeholder - map lane tensor when enabled.
            agent_types: (batch, agents, num_agent_types) - one-hot agent type features, ego first.

        Output:
            out_dists: (modes, future_steps, batch, agents, 5 or 6) - joint trajectory distribution parameters.
            mode_probs: (batch, modes) - normalized latent mode probabilities.

"""
        pass


# --- [Original file: utils/train_helpers.py] ---
def get_BVG_distributions_joint(pred):
    B = pred.size(0)
    T = pred.size(1)
    N = pred.size(2)
    mu_x = pred[:, :, :, 0].unsqueeze(3)
    mu_y = pred[:, :, :, 1].unsqueeze(3)
    sigma_x = pred[:, :, :, 2]
    sigma_y = pred[:, :, :, 3]
    rho = pred[:, :, :, 4]

    cov = torch.zeros((B, T, N, 2, 2)).to(pred.device)
    cov[:, :, :, 0, 0] = sigma_x ** 2
    cov[:, :, :, 1, 1] = sigma_y ** 2
    cov_val = rho * sigma_x * sigma_y
    cov[:, :, :, 0, 1] = cov_val
    cov[:, :, :, 1, 0] = cov_val

    biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
    return biv_gauss_dist


def get_Laplace_dist_joint(pred):
    return Laplace(pred[:, :, :, :2], pred[:, :, :, 2:4])


def nll_pytorch_dist_joint(pred, data, agents_masks):
    # biv_gauss_dist = get_BVG_distributions_joint(pred)
    biv_gauss_dist = get_Laplace_dist_joint(pred)
    num_active_agents_per_timestep = agents_masks.sum(2)
    loss = (((-biv_gauss_dist.log_prob(data).sum(-1) * agents_masks).sum(2)) / num_active_agents_per_timestep).sum(1)
    return loss


def nll_loss_multimodes_joint(pred, ego_data, agents_data, mode_probs, entropy_weight=1.0, kl_weight=1.0,
                              use_FDEADE_aux_loss=True, agent_types=None, predict_yaw=False):
    """
    [TODO] Compute the AutoBot-Joint multimodal latent-variable training objective.

    Input:
        pred: (modes, future_steps, batch, agents, 5 or 6) - predicted joint distribution parameters.
        ego_data: (batch, future_steps, features) - ego future ground truth.
        agents_data: (batch, future_steps, other_agents, features) - other-agent future ground truth and masks.
        mode_probs: (batch, modes) - prior probabilities over latent modes.
        entropy_weight: float - scale for distribution entropy regularization.
        kl_weight: float - scale for prior/posterior KL term.

    Output:
        loss: scalar tensor - posterior-weighted joint negative log likelihood plus entropy regularization.
        kl_loss: scalar tensor - weighted KL divergence between prior and posterior mode probabilities.
        post_entropy: float - mean posterior entropy.
        adefde_loss: scalar tensor - optional best-mode ADE/FDE auxiliary loss.

"""
    pass


def l2_loss_fde_joint(pred, data, agent_masks, agent_types, predict_yaw):
    fde_loss = (torch.norm((pred[:, -1, :, :, :2].transpose(0, 1) - data[:, -1, :, :2].unsqueeze(1)), 2, dim=-1) *
                agent_masks[:, -1:, :]).mean(-1)
    ade_loss = (torch.norm((pred[:, :, :, :, :2].transpose(1, 2) - data[:, :, :, :2].unsqueeze(0)), 2, dim=-1) *
                agent_masks.unsqueeze(0)).mean(-1).mean(dim=2).transpose(0, 1)

    yaw_loss = torch.tensor(0.0).to(pred.device)
    if predict_yaw:
        vehicles_only = (agent_types[:, :, 0] == 1.0).unsqueeze(0)
        yaw_loss = torch.norm(pred[:, :, :, :, 5:].transpose(1, 2) - data[:, :, :, 4:5].unsqueeze(0), dim=-1).mean(2)  # across time
        yaw_loss = (yaw_loss * vehicles_only).mean(-1).transpose(0, 1)

    loss, min_inds = (fde_loss + ade_loss + yaw_loss).min(dim=1)
    return 100.0 * loss.mean()


# ============================================================
# __main__: Automated test suite for 9 ablated functions
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    passed = 0
    failed = 0

    def check(test_name, condition, detail=""):
        global passed, failed
        if bool(condition):
            passed += 1
            print(f"  [{test_name}] PASS")
        else:
            failed += 1
            suffix = f" - {detail}" if detail else ""
            print(f"  [{test_name}] FAIL{suffix}")

    def skip_checks(count, reason):
        global failed
        failed += count
        print(f"  [skipped {count} check(s)] FAIL - {reason}")

    device = torch.device("cpu")

    def build_joint(use_map_lanes=False):
        return AutoBotJoint(
            d_k=16, _M=2, c=3, T=4, L_enc=1, dropout=0.0, k_attr=2, map_attr=3,
            num_heads=4, L_dec=1, tx_hidden_size=32, use_map_lanes=use_map_lanes,
            num_agent_types=3, predict_yaw=False
        ).to(device).eval()

    print("=" * 70)
    print("AutoBots: AutoBot-Joint Core Model Benchmark")
    print("Automated Test Suite - 9 ablated functions")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    print("-" * 60)
    print("[Test 1/9] OutputModel.forward - stable bivariate distribution parameters")
    try:
        module = OutputModel(d_k=16, predict_yaw=True).to(device).eval()
        latent = torch.randn(4, 6, 16, device=device, requires_grad=True)
        out = module(latent)
        check("OutputModel output not None", out is not None)
        if out is not None:
            check("OutputModel output shape", out.shape == (4, 6, 6), f"got {tuple(out.shape)}")
            check("OutputModel finite", torch.isfinite(out).all().item())
            check("OutputModel sigma positive", (out[:, :, 2:4] > module.min_stdev).all().item())
            check("OutputModel rho bounded", (out[:, :, 4].abs() <= 0.900001).all().item())
            out.sum().backward()
            check("OutputModel gradient", latent.grad is not None and torch.isfinite(latent.grad).all().item())
        else:
            skip_checks(5, "OutputModel.forward returned None")
    except Exception as exc:
        skip_checks(6, f"OutputModel.forward raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 2/9] MapEncoderPtsMA.get_road_pts_mask - empty road safety")
    try:
        roads = torch.ones(2, 3, 4, 5, 4, device=device)
        roads[0, 1, :, :, -1] = 0.0
        roads[1, 2, 3, :, -1] = 0.0
        segment_mask, point_mask = MapEncoderPtsMA(d_k=16, map_attr=3, dropout=0.0).get_road_pts_mask(roads)
        check("Map mask output not None", segment_mask is not None and point_mask is not None)
        if segment_mask is not None and point_mask is not None:
            check("Road segment mask shape", segment_mask.shape == (2, 3, 4), f"got {tuple(segment_mask.shape)}")
            check("Road point mask shape", point_mask.shape == (2 * 3 * 4, 5), f"got {tuple(point_mask.shape)}")
            check("Empty road first segment unmasked", segment_mask[0, 1, 0].item() is False)
            check("All-empty point rows have one unmasked point", (point_mask.sum(-1) < 5).all().item())
        else:
            skip_checks(4, "MapEncoderPtsMA.get_road_pts_mask returned None")
    except Exception as exc:
        skip_checks(5, f"MapEncoderPtsMA.get_road_pts_mask raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 3/9] MapEncoderPtsMA.forward - seeded lane set encoding")
    try:
        module = MapEncoderPtsMA(d_k=16, map_attr=3, dropout=0.0).to(device).eval()
        roads = torch.randn(2, 3, 4, 5, 4, device=device)
        roads[:, :, :, :, -1] = 1.0
        roads[0, 2, :, :, -1] = 0.0
        agents_emb = torch.randn(3, 2, 3, 16, device=device)
        feats, mask = module(roads, agents_emb)
        check("MapEncoder output not None", feats is not None and mask is not None)
        if feats is not None and mask is not None:
            check("MapEncoder feature shape", feats.shape == (4, 2, 3, 16), f"got {tuple(feats.shape)}")
            check("MapEncoder mask shape", mask.shape == (2, 3, 4), f"got {tuple(mask.shape)}")
            check("MapEncoder finite", torch.isfinite(feats).all().item())
            check("MapEncoder empty road protected", mask[0, 2, 0].item() is False)
        else:
            skip_checks(4, "MapEncoderPtsMA.forward returned None")
    except Exception as exc:
        skip_checks(5, f"MapEncoderPtsMA.forward raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 4/9] AutoBotJoint.temporal_attn_fn - per-agent temporal encoding")
    try:
        model = build_joint()
        agents_emb = torch.randn(3, 2, 3, 16, device=device, requires_grad=True)
        masks = torch.zeros(2, 3, 3, dtype=torch.bool, device=device)
        masks[0, :, 2] = True
        out = model.temporal_attn_fn(agents_emb, masks, model.temporal_attn_layers[0])
        check("Temporal encoder output not None", out is not None)
        if out is not None:
            check("Temporal encoder shape", out.shape == (3, 2, 3, 16), f"got {tuple(out.shape)}")
            check("Temporal encoder finite", torch.isfinite(out).all().item())
            out.sum().backward()
            check("Temporal encoder gradient", agents_emb.grad is not None and torch.isfinite(agents_emb.grad).all().item())
        else:
            skip_checks(3, "temporal_attn_fn returned None")
    except Exception as exc:
        skip_checks(4, f"temporal_attn_fn raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 5/9] AutoBotJoint.social_attn_fn - cross-agent set encoding")
    try:
        model = build_joint()
        agents_emb = torch.randn(3, 2, 3, 16, device=device, requires_grad=True)
        masks = torch.zeros(2, 3, 3, dtype=torch.bool, device=device)
        masks[1, :, 1] = True
        out = model.social_attn_fn(agents_emb, masks, model.social_attn_layers[0])
        check("Social encoder output not None", out is not None)
        if out is not None:
            check("Social encoder shape", out.shape == (3, 2, 3, 16), f"got {tuple(out.shape)}")
            check("Social encoder finite", torch.isfinite(out).all().item())
            out.sum().backward()
            check("Social encoder gradient", agents_emb.grad is not None and torch.isfinite(agents_emb.grad).all().item())
        else:
            skip_checks(3, "social_attn_fn returned None")
    except Exception as exc:
        skip_checks(4, f"social_attn_fn raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 6/9] AutoBotJoint.temporal_attn_decoder_fn - causal multimodal decoder")
    try:
        model = build_joint()
        agents_emb = torch.randn(4, 6, 3, 16, device=device, requires_grad=True)
        context = torch.randn(3, 6, 3, 16, device=device)
        masks = torch.zeros(6, 3, 3, dtype=torch.bool, device=device)
        masks[0, :, 2] = True
        out = model.temporal_attn_decoder_fn(agents_emb, context, masks, model.temporal_attn_decoder_layers[0])
        decoder_mask = model.generate_decoder_mask(4, device)
        check("Temporal decoder output not None", out is not None)
        if out is not None:
            check("Temporal decoder shape", out.shape == (4, 6, 3, 16), f"got {tuple(out.shape)}")
            check("Temporal decoder finite", torch.isfinite(out).all().item())
            check("Decoder causal mask", decoder_mask[0, 1].item() and not decoder_mask[1, 0].item())
            out.sum().backward()
            check("Temporal decoder gradient", agents_emb.grad is not None and torch.isfinite(agents_emb.grad).all().item())
        else:
            skip_checks(4, "temporal_attn_decoder_fn returned None")
    except Exception as exc:
        skip_checks(5, f"temporal_attn_decoder_fn raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 7/9] AutoBotJoint.social_attn_decoder_fn - decoder cross-agent attention")
    try:
        model = build_joint()
        agents_emb = torch.randn(4, 6, 3, 16, device=device, requires_grad=True)
        masks = torch.zeros(6, 3, 3, dtype=torch.bool, device=device)
        masks[:, -1, 2] = True
        out = model.social_attn_decoder_fn(agents_emb, masks, model.social_attn_decoder_layers[0])
        check("Social decoder output not None", out is not None)
        if out is not None:
            check("Social decoder shape", out.shape == (4, 6, 3, 16), f"got {tuple(out.shape)}")
            check("Social decoder finite", torch.isfinite(out).all().item())
            out.sum().backward()
            check("Social decoder gradient", agents_emb.grad is not None and torch.isfinite(agents_emb.grad).all().item())
        else:
            skip_checks(3, "social_attn_decoder_fn returned None")
    except Exception as exc:
        skip_checks(4, f"social_attn_decoder_fn raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 8/9] AutoBotJoint.forward - joint multimodal trajectory prediction")
    try:
        model = build_joint()
        ego = torch.randn(2, 3, 3, device=device)
        ego[:, :, -1] = 1.0
        agents = torch.randn(2, 3, 2, 3, device=device)
        agents[:, :, :, -1] = 1.0
        roads = torch.zeros(2, 1, 1, device=device)
        agent_types = torch.zeros(2, 3, 3, device=device)
        agent_types[:, :, 0] = 1.0
        pred, probs = model(ego, agents, roads, agent_types)
        check("AutoBotJoint output not None", pred is not None and probs is not None)
        if pred is not None and probs is not None:
            check("AutoBotJoint pred shape", pred.shape == (3, 4, 2, 3, 5), f"got {tuple(pred.shape)}")
            check("AutoBotJoint mode shape", probs.shape == (2, 3), f"got {tuple(probs.shape)}")
            check("AutoBotJoint outputs finite", torch.isfinite(pred).all().item() and torch.isfinite(probs).all().item())
            check("AutoBotJoint sigmas positive", (pred[:, :, :, :, 2:4] > model.output_model.min_stdev).all().item())
            check("AutoBotJoint mode probabilities sum", torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5))
        else:
            skip_checks(5, "AutoBotJoint.forward returned None")
    except Exception as exc:
        skip_checks(6, f"AutoBotJoint.forward raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 9/9] nll_loss_multimodes_joint - latent posterior multimode objective")
    try:
        pred = torch.zeros(3, 4, 2, 3, 5, device=device)
        pred[..., 2:4] = 0.8
        pred[..., 4] = 0.0
        pred = pred.requires_grad_()
        ego_data = torch.zeros(2, 4, 5, device=device)
        agents_data = torch.zeros(2, 4, 2, 5, device=device)
        agents_data[..., -1] = 1.0
        mode_probs = torch.full((2, 3), 1.0 / 3.0, device=device, requires_grad=True)
        loss, kl_loss, post_entropy, aux_loss = nll_loss_multimodes_joint(
            pred, ego_data, agents_data, mode_probs,
            entropy_weight=0.1, kl_weight=0.5, use_FDEADE_aux_loss=True,
            agent_types=None, predict_yaw=False
        )
        check("Joint NLL outputs not None", loss is not None and kl_loss is not None and aux_loss is not None)
        if loss is not None and kl_loss is not None and aux_loss is not None:
            check("Joint NLL scalars", loss.dim() == 0 and kl_loss.dim() == 0 and aux_loss.dim() == 0)
            check("Joint NLL finite", torch.isfinite(loss).item() and torch.isfinite(kl_loss).item() and torch.isfinite(aux_loss).item())
            check("Joint posterior entropy finite", np.isfinite(post_entropy))
            (loss + kl_loss + aux_loss).backward()
            check("Joint NLL gradients", pred.grad is not None and torch.isfinite(pred.grad).all().item())
        else:
            skip_checks(4, "nll_loss_multimodes_joint returned None")
    except Exception as exc:
        skip_checks(5, f"nll_loss_multimodes_joint raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
