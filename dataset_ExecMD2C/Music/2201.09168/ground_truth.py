"""Ground-truth core model components for the RIVRL benchmark.

This file consolidates only the architecture-level components for Reading-strategy
Inspired Visual Representation Learning: previewing-aware attention, video/text
multi-level encoders, latent/hybrid mapping modules, and the retrieval triplet loss.
"""

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# --- [Original file: loss.py] ---

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def euclidean_sim(im, s):
    """L2 distance
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.pow(2).sum(2).t()
    return score


def L1_sim(im, s):
    """L1 distance
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.abs().sum(2).t()
    return score


def L1_sim_norm(im, s):
    """L1 normalization distance  1 - L_1/K
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = YmX.abs().sum(2).t()/im.size(1) -1
    return score


def L2_sim(im, s):
    """L2 distance
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.pow(2).sum(2).t()
    return score


def L2_sim_norm(im, s):
    """L2 normalization distance  1 - L_2/K
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = YmX.pow(2).sum(2).t()/im.size(1) - 1
    return score


def jaccard_sim(im, s):
    im_bs = im.size(0)
    s_bs = s.size(0)
    im = im.unsqueeze(1).expand(-1,s_bs,-1)
    s = s.unsqueeze(0).expand(im_bs,-1,-1)
    intersection = torch.min(im,s).sum(-1)
    union = torch.max(im,s).sum(-1)
    score = intersection / union
    return score


NAME_TO_SIM = {'cosine': cosine_sim, 'order': order_sim, 'euclidean': euclidean_sim, 'jaccard': jaccard_sim}


def get_sim(name):
    assert name in NAME_TO_SIM, '%s not supported.'%name
    return NAME_TO_SIM[name]


class TripletLoss(nn.Module):
    """
    triplet ranking loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, cost_style='sum', direction='all'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        elif measure == 'jaccard':
            self.sim = jaccard_sim
        elif measure == 'l1':
            self.sim = L1_sim
        elif measure == 'l2':
            self.sim = L2_sim
        elif measure == 'l1_norm':
            self.sim = L1_sim_norm
        elif measure == 'l2_norm':
            self.sim = L2_sim_norm
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, s, im):
        # compute video-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['v2t', 'all']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2v', 'all']:
            # video retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = Variable(torch.zeros(1)).cuda()
        if cost_im is None:
            cost_im = Variable(torch.zeros(1)).cuda()

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()


# --- [Original file: PaA.py] ---

class Previewing_aware_Attention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, opt, dim_in_q, dim_in, dim_k, dim_v, dropout=0.2):
        super(Previewing_aware_Attention, self).__init__()
        self.dim_in_q = dim_in_q
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(self.dim_in_q, self.dim_k, bias=False)
        self.linear_k = nn.Linear(self.dim_in, self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_in, self.dim_v, bias=False)
        self.fc_v = nn.Linear(self.dim_v, self.dim_in)
        self.layer_norm = nn.LayerNorm(dim_v, eps=1e-6)
        self._norm_fact = 1 / math.sqrt(self.dim_k)
        self.dropout = nn.Dropout(dropout)
        self.pooling = opt.pooling

    def forward(self, mask, x, y):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        if self.pooling == 'mean':
            residual = F.avg_pool1d(x.permute(0, 2, 1), x.size(1)).squeeze(2)
        else:
            residual = F.max_pool1d(x.permute(0, 2, 1), x.size(1)).squeeze(2)

        q = self.linear_q(y)
        k = self.linear_k(x)
        v = self.linear_v(x)

        attention_mask = mask.unsqueeze(1)
        attention_mask = (1.0 - attention_mask) * -10000.0  # padding鐨則oken缃负-10000锛宔xp(-1w)=0

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact

        attention_scores = dist + attention_mask
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        att = torch.bmm(attention_probs, v).squeeze(1)
        att = self.layer_norm(self.fc_v(att) + residual)

        return att


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        del residual

        x = self.layer_norm(x)

        return x


class qkv_layer(nn.Module):
    def __init__(self, opt, q_input_dim, kv_input_dim, qkv_out_dim):
        super(qkv_layer, self).__init__()
        self.qk_dim = 512
        self.qkv = Previewing_aware_Attention(opt, q_input_dim, kv_input_dim, self.qk_dim, qkv_out_dim)
        self.ffn = PositionwiseFeedForward(qkv_out_dim, qkv_out_dim * 2)

    def forward(self, mask, kv_data, q_data):

        qkv_out = self.qkv(mask, kv_data, q_data)
        qkv_out = self.ffn(qkv_out)

        return qkv_out


# --- [Original file: model.py] ---

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                              fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)


def get_mask(kernel_size, stride, lengths, mask):
    lengths = torch.tensor(lengths)
    # num = (lengths / stride).int()
    num = ((lengths - kernel_size) / stride).int() + 1
    conv_len = math.floor((mask.size(1) - kernel_size) / stride) + 1
    mask_change = torch.zeros(mask.size(0), conv_len)
    for k in range(mask.size(0)):
        # if num[k] > conv_len:
        #     num[k] = conv_len
        mask_change[k, :num[k]] = 1.0
    return mask_change.cuda()


class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """

    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch normalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features


class Video_preview_intensive_encoding(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Video_preview_intensive_encoding, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.gru_pool = opt.gru_pool
        self.space = opt.space

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        self.num_cnn = opt.num_cnn
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (opt.visual_kernel_sizes[i], 2048), stride=opt.visual_kernel_stride[i])
            for i in range(self.num_cnn)
        ])

        self.kernel_sizes = opt.visual_kernel_sizes
        self.stride = opt.visual_kernel_stride

        self.fc_org = nn.Linear(opt.visual_feat_dim, 2048)

        self.paa = qkv_layer(opt, self.rnn_output_size, opt.qkv_input_dim, opt.qkv_out_dim)

        if opt.space == 'latent':
            self.vid_mapping_preview = Latent_mapping(opt.visual_mapping_layers_preview,
                                                     opt.dropout, opt.tag_vocab_size).cuda()
            self.vid_mapping_intensive = Latent_mapping(opt.visual_mapping_layers_intensive,
                                                    opt.dropout, opt.tag_vocab_size).cuda()
        else:
            self.vid_mapping_preview = Hybrid_mapping(opt.visual_mapping_layers_preview,
                                                     opt.dropout, opt.tag_vocab_size).cuda()
            self.vid_mapping_intensive = Hybrid_mapping(opt.visual_mapping_layers_intensive,
                                                    opt.dropout, opt.tag_vocab_size).cuda()


    def forward(self, videos):
        """Extract video feature vectors."""
        # self.rnn.flatten_parameters()
        videos, videos_origin, lengths, mask = videos
        del videos_origin

        # previewing_branch
        gru_init_out, _ = self.rnn(videos)
        if self.gru_pool == 'mean':
            mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
            for i, batch in enumerate(gru_init_out):
                mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
            gru_out = mean_gru
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, mask.unsqueeze(-1)), 1)[0]
        preview_out = self.dropout(gru_out)

        # intensive-reading_branch
        # Map to lower dimensions
        con_input = F.relu(self.fc_org(videos))

        mask_con = mask.unsqueeze(2).expand(-1, -1, con_input.size(2))  # (N,C,F1)
        con_input_mask = con_input * mask_con
        con_input = con_input_mask.unsqueeze(1)
        con_out_list = []
        for i in range(self.num_cnn):
            con_out_i = F.relu(self.convs1[i](con_input)).squeeze(3).permute(0, 2, 1)
            con_out_list.append(con_out_i)
        del mask_con, con_input

        # previewing-aware attention
        intensive_out_list = []
        aware_out_frame = self.paa(mask, con_input_mask, preview_out.unsqueeze(1))
        intensive_out_list.append(aware_out_frame)
        for i in range(self.num_cnn):
            aware_mask = get_mask(self.kernel_sizes[i], self.stride[i], lengths, mask)
            aware_out_i = self.paa(aware_mask, con_out_list[i], preview_out.unsqueeze(1))
            intensive_out_list.append(aware_out_i)

        intensive_out = torch.cat(intensive_out_list, 1)
        del mask, con_out_list, intensive_out_list

        # mapping--
        if self.space == 'latent':
            preview_out = self.vid_mapping_preview(preview_out)
            intensive_out = self.vid_mapping_intensive(intensive_out)
        else:
            preview_out, preview_concept = self.vid_mapping_preview(preview_out)
            intensive_out, intensive_concept = self.vid_mapping_intensive(intensive_out)

        if self.space == 'latent':
            return preview_out, intensive_out
        else:
            return (preview_out, preview_concept), (intensive_out, intensive_concept)

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
                # print(new_state[name], ':', param)

        super(Video_preview_intensive_encoding, self).load_state_dict(new_state)


class Text_multilevel_encoding(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.gru_pool = opt.gru_pool
        self.loss_fun = opt.loss_fun

        # visual bidirectional rnn encoder
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size),
                      padding=(int((window_size - 1) / 2), 0))
            for window_size in opt.text_kernel_sizes
        ])

        if opt.space == 'latent':
            self.text_mapping_preview = Latent_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size).cuda()
            self.text_mapping_intensive = Latent_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size).cuda()
        else:
            self.text_mapping_preview = Hybrid_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size).cuda()
            self.text_mapping_intensive = Hybrid_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size).cuda()

        self.init_weights()

        self.space = opt.space

        self.use_bert = opt.use_bert

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, text, *args):
        # Embed word ids to vectors
        self.rnn.flatten_parameters()

        cap_wids, cap_bows, cap_bert, lengths, cap_mask = text

        # Level 1. Global Encoding by Mean Pooling According
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU
        cap_wids = self.embed(cap_wids)
        packed = pack_padded_sequence(cap_wids, lengths, batch_first=True)
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]

        if self.gru_pool == 'mean':
            gru_out = Variable(torch.zeros(padded[0].size(0), self.rnn_output_size)).cuda()
            for i, batch in enumerate(padded[0]):
                gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, cap_mask.unsqueeze(-1)), 1)[0]
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        if self.use_bert == 1:
            features = torch.cat((gru_out, con_out, org_out, cap_bert), 1)
        else:
            features = torch.cat((gru_out, con_out, org_out), 1)

        if self.space == 'latent':
            features_list = []
            features_preview = self.text_mapping_preview(features)
            features_intensive = self.text_mapping_intensive(features)
            features_list.append(features_preview)
            features_list.append(features_intensive)
        else:
            features_preview, features_caption_preview = self.text_mapping_preview(features)
            features_intensive, features_caption_intensive = self.text_mapping_intensive(features)

        if self.space == 'latent':
            return features_list
        else:
            return (features_preview, features_caption_preview), (features_intensive, features_caption_intensive)


class Hybrid_mapping(nn.Module):

    def __init__(self, mapping_layers, dropout, tag_vocab_size, l2norm=True):
        super(Hybrid_mapping, self).__init__()

        self.l2norm = l2norm
        self.mapping = MFC(mapping_layers, dropout, have_bn=True, have_last_bn=True)

        self.tag_fc = nn.Linear(mapping_layers[0], tag_vocab_size)
        self.tag_fc_batch_norm = nn.BatchNorm1d(tag_vocab_size)

    def forward(self, features):
        # mapping to concept space
        tag_prob = self.tag_fc(features)
        tag_prob = self.tag_fc_batch_norm(tag_prob)
        concept_features = torch.sigmoid(tag_prob)

        # mapping to latent space
        latent_features = self.mapping(features)
        if self.l2norm:
            latent_features = l2norm(latent_features)

        return (latent_features, concept_features)


class Latent_mapping(nn.Module):

    def __init__(self, mapping_layers, dropout, l2norm=True):
        super(Latent_mapping, self).__init__()

        self.l2norm = l2norm
        self.mapping = MFC(mapping_layers, dropout, have_bn=True, have_last_bn=True)

    def forward(self, features):
        # mapping to latent space
        latent_features = self.mapping(features)
        if self.l2norm:
            latent_features = l2norm(latent_features)

        return latent_features


if __name__ == "__main__":
    torch.manual_seed(42)

    # Keep source model code intact above; make the synthetic tests CPU-friendly.
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.Tensor.cuda = lambda self, *args, **kwargs: self

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

    class Opt:
        pass

    def make_opt():
        opt = Opt()
        opt.pooling = "mean"
        opt.dropout = 0.0
        opt.visual_feat_dim = 8
        opt.visual_rnn_size = 4
        opt.visual_norm = False
        opt.gru_pool = "mean"
        opt.space = "latent"
        opt.num_cnn = 0
        opt.visual_kernel_num = 2048
        opt.visual_kernel_sizes = []
        opt.visual_kernel_stride = []
        opt.qkv_input_dim = 2048
        opt.qkv_out_dim = 2048
        opt.visual_mapping_layers_preview = [8, 6]
        opt.visual_mapping_layers_intensive = [2048, 6]
        opt.tag_vocab_size = 5
        opt.word_dim = 6
        opt.we_parameter = None
        opt.text_rnn_size = 4
        opt.loss_fun = "mrl"
        opt.vocab_size = 20
        opt.text_kernel_num = 3
        opt.text_kernel_sizes = [3]
        opt.text_mapping_layers = [8 + 3 + 7, 6]
        opt.use_bert = 0
        return opt

    print("[Test 1/5] Previewing_aware_Attention.forward")
    try:
        opt = make_opt()
        module = Previewing_aware_Attention(opt, dim_in_q=4, dim_in=6, dim_k=5, dim_v=6, dropout=0.0)
        module.eval()
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float32)
        x = torch.randn(2, 4, 6, requires_grad=True)
        y = torch.randn(2, 1, 4, requires_grad=True)
        out = module(mask, x, y)
        check("PaA output not None", out is not None)
        if out is not None:
            check("PaA output shape", tuple(out.shape) == (2, 6), str(tuple(out.shape)))
            check("PaA finite output", torch.isfinite(out).all().item())
            out.sum().backward()
            check("PaA gradient reaches inputs", x.grad is not None and x.grad.abs().sum().item() > 0 and y.grad is not None and y.grad.abs().sum().item() > 0)
            check("PaA keeps configured scale", abs(module._norm_fact - (1 / math.sqrt(5))) < 1e-8)
        else:
            skip_checks(4, "PaA returned None")
    except Exception as exc:
        skip_checks(5, f"Previewing_aware_Attention.forward raised {type(exc).__name__}: {exc}")

    print("[Test 2/5] qkv_layer.forward")
    try:
        opt = make_opt()
        module = qkv_layer(opt, q_input_dim=4, kv_input_dim=6, qkv_out_dim=6)
        module.eval()
        mask = torch.ones(2, 4)
        kv = torch.randn(2, 4, 6, requires_grad=True)
        q = torch.randn(2, 1, 4, requires_grad=True)
        out = module(mask, kv, q)
        check("qkv_layer output not None", out is not None)
        if out is not None:
            check("qkv_layer output shape", tuple(out.shape) == (2, 6), str(tuple(out.shape)))
            check("qkv_layer finite output", torch.isfinite(out).all().item())
            out.pow(2).sum().backward()
            check("qkv_layer gradient reaches inputs", kv.grad is not None and kv.grad.abs().sum().item() > 0 and q.grad is not None and q.grad.abs().sum().item() > 0)
            check("qkv_layer has FFN refinement", isinstance(module.ffn, PositionwiseFeedForward))
        else:
            skip_checks(4, "qkv_layer returned None")
    except Exception as exc:
        skip_checks(5, f"qkv_layer.forward raised {type(exc).__name__}: {exc}")

    print("[Test 3/5] Video_preview_intensive_encoding.forward")
    try:
        opt = make_opt()
        model = Video_preview_intensive_encoding(opt)
        model.eval()
        videos = torch.randn(2, 4, 8)
        videos_origin = torch.zeros(2, 8)
        lengths = [4, 3]
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.float32)
        preview, intensive = model((videos, videos_origin, lengths, mask))
        check("Video encoder outputs not None", preview is not None and intensive is not None)
        if preview is not None and intensive is not None:
            check("Video preview shape", tuple(preview.shape) == (2, 6), str(tuple(preview.shape)))
            check("Video intensive shape", tuple(intensive.shape) == (2, 6), str(tuple(intensive.shape)))
            check("Video outputs finite", torch.isfinite(preview).all().item() and torch.isfinite(intensive).all().item())
            check("Video outputs L2-normalized", torch.allclose(preview.norm(dim=1), torch.ones(2), atol=1e-4) and torch.allclose(intensive.norm(dim=1), torch.ones(2), atol=1e-4))
        else:
            skip_checks(4, "Video encoder returned None")
    except Exception as exc:
        skip_checks(5, f"Video_preview_intensive_encoding.forward raised {type(exc).__name__}: {exc}")

    print("[Test 4/5] Text_multilevel_encoding.forward")
    try:
        opt = make_opt()
        model = Text_multilevel_encoding(opt)
        model.eval()
        cap_wids = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]], dtype=torch.long)
        cap_bows = torch.randn(2, 7)
        cap_bert = None
        lengths = [5, 3]
        cap_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
        outputs = model((cap_wids, cap_bows, cap_bert, lengths, cap_mask))
        check("Text encoder outputs not None", outputs is not None)
        if outputs is not None:
            check("Text encoder returns two levels", isinstance(outputs, list) and len(outputs) == 2)
            check("Text output shapes", tuple(outputs[0].shape) == (2, 6) and tuple(outputs[1].shape) == (2, 6))
            check("Text outputs finite", torch.isfinite(outputs[0]).all().item() and torch.isfinite(outputs[1]).all().item())
            check("Text outputs L2-normalized", torch.allclose(outputs[0].norm(dim=1), torch.ones(2), atol=1e-4) and torch.allclose(outputs[1].norm(dim=1), torch.ones(2), atol=1e-4))
        else:
            skip_checks(4, "Text encoder returned None")
    except Exception as exc:
        skip_checks(5, f"Text_multilevel_encoding.forward raised {type(exc).__name__}: {exc}")

    print("[Test 5/5] TripletLoss.forward")
    try:
        loss_fn = TripletLoss(margin=0.2, measure="cosine", max_violation=True, cost_style="sum", direction="all")
        s = F.normalize(torch.eye(3), dim=1)
        im = F.normalize(torch.tensor([[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.1, 0.0, 1.0]]), dim=1)
        loss = loss_fn(s, im)
        check("TripletLoss output not None", loss is not None)
        if loss is not None:
            check("TripletLoss scalar shape", getattr(loss, "shape", torch.Size([])) == torch.Size([]))
            check("TripletLoss finite", torch.isfinite(loss).item())
            check("TripletLoss nonnegative", loss.item() >= 0)
            loss_mean = TripletLoss(margin=0.2, measure="cosine", max_violation=False, cost_style="mean", direction="t2v")(s, im)
            check("TripletLoss supports mean/t2v branch", loss_mean is not None and torch.isfinite(loss_mean).item())
        else:
            skip_checks(4, "TripletLoss returned None")
    except Exception as exc:
        skip_checks(5, f"TripletLoss.forward raised {type(exc).__name__}: {exc}")

    print(f"RESULT: passed={passed} failed={failed}")
    if failed != 0:
        raise SystemExit(1)
