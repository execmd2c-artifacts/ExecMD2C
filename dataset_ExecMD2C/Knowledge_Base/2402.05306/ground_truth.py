"""
Self-contained Sym-Q benchmark answer key.

This file consolidates the core model components from the Sym-Q repository:
Set Transformer attention blocks, the point-set encoder, the expression-tree
encoder, the SymQ action-value head, and the supervised contrastive loss used
during training. Reinforcement-learning memory/agent code, beam search,
symbolic environments, dataset wrappers, BFGS post-processing, evaluation
scripts, checkpoint I/O, and training loops are intentionally excluded.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# --- [Original file: attention_block.py] ---

class MAB(nn.Module):
    """
    Multi-Headed Attention Block (MAB).

    Implements the multi-head attention mechanism, which is a key component of the Transformer architecture.

    Attributes:
        dim_V (int): Dimension of the value.
        num_heads (int): Number of attention heads.
        fc_q (nn.Linear): Fully connected layer for transforming the query input.
        fc_k (nn.Linear): Fully connected layer for transforming the key input.
        fc_v (nn.Linear): Fully connected layer for transforming the value input.
        ln0 (nn.LayerNorm, optional): Layer normalization applied before the output linear layer.
        ln1 (nn.LayerNorm, optional): Layer normalization applied after the output linear layer.
        fc_o (nn.Linear): Output fully connected layer.
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        """
        Initialize the MAB.

        Args:
            dim_Q (int): Dimension of the query input.
            dim_K (int): Dimension of the key input.
            dim_V (int): Dimension of the value input.
            num_heads (int): Number of attention heads.
            ln (bool, optional): Whether to use layer normalization. Default is False.
        """
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        """
        Forward pass for MAB.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_length, dim_Q).
            K (torch.Tensor): Key tensor of shape (batch_size, seq_length, dim_K).

        Returns:
            torch.Tensor: Output tensor after multi-head attention mechanism.
        """

        # Transform inputs using fully connected layers
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        # Split tensors for multi-head attention
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # Compute attention scores
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)

        # Compute weighted sum of values
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        # Apply layer normalization if enabled
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)

        # Add a residual connection with a linear transformation and ReLU activation
        O = O + F.relu(self.fc_o(O))

        # Apply another layer normalization if enabled
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)

        return O


class ISAB(nn.Module):
    """
    Induced Set Attention Block (ISAB).

    Implements a mechanism where the attention is not computed for every pair of points, but instead,
    it's induced through a set of learnable inducing points, thereby reducing computational complexity.

    Attributes:
        I (nn.Parameter): A set of learnable inducing points.
        mab0 (MAB): First Multi-Headed Attention Block.
        mab1 (MAB): Second Multi-Headed Attention Block.
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        """
        Initialize the ISAB.

        Args:
            dim_in (int): Dimension of the input.
            dim_out (int): Desired dimension of the output.
            num_heads (int): Number of attention heads for the MABs.
            num_inds (int): Number of inducing points.
            ln (bool, optional): Whether to use layer normalization in MABs. Default is False.
        """
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        # Define the two Multi-Headed Attention Blocks
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        """
        Forward pass for ISAB.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_length, dim_in).

        Returns:
            torch.Tensor: Output tensor after inducing set attention mechanism.
        """

        # Compute the induced attention from the learnable inducing points to the input X
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)

        # Compute the attention from the input X to the induced attention H
        return self.mab1(X, H)


class PMA(nn.Module):
    """
    Pooling by Multi-Headed Attention (PMA).

    Implements a pooling mechanism that uses a set of learnable seed vectors,
    which are attended over by the input set, allowing for a form of set reduction.

    Attributes:
        S (nn.Parameter): A set of learnable seed vectors.
        mab (MAB): Multi-Headed Attention Block used for the pooling.
    """

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        """
        Initialize the PMA.

        Args:
            dim (int): Dimension of the input and the seed vectors.
            num_heads (int): Number of attention heads for the MAB.
            num_seeds (int): Number of seed vectors.
            ln (bool, optional): Whether to use layer normalization in the MAB. Default is False.
        """
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)

        # Define the Multi-Headed Attention Block for pooling
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        """
        Forward pass for PMA.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_length, dim).

        Returns:
            torch.Tensor: Pooled output tensor after attending over the learnable seed vectors.
        """

        # Attend over the learnable seed vectors using the input set X
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# --- [Original file: encoder.py] ---

class SetEncoder(pl.LightningModule):
    """
    Set Encoder Module.

    A neural network model designed for encoding sets into a fixed-sized representation.
    This model provides the flexibility to choose different data representations,
    normalization strategies, and neural architectures.

    Attributes:
        linear (bool): Flag to use a linear transformation.
        bit16 (bool): Flag to represent input data in 16-bit format.
        norm (bool): Flag for normalization.
        activation (str): Type of activation function after the linear transformation.
        input_normalization (bool): Flag to normalize the input data.
        linearl (nn.Linear): Linear transformation layer.
        selfatt (nn.ModuleList): List of ISAB (Induced Set Attention Blocks) layers.
        selfatt1 (ISAB): Initial ISAB layer.
        outatt (PMA): Point-wise Multihead Attention layer.
        _device (torch.device): The device (CPU or GPU) where the module is deployed.
    """

    def __init__(self, cfg):
        """
        Initialize the SetEncoder.

        Args:
            cfg (dict): Configuration dictionary containing model hyperparameters.
        """
        super(SetEncoder, self).__init__()
        self.linear = cfg["linear"]
        self.bit16 = cfg["bit16"]
        self.norm = cfg["norm"]
        assert (
            cfg["linear"] != cfg["bit16"]
        ), "one and only one between linear and bit16 must be true at the same time"
        if cfg["norm"]:
            self.register_buffer("mean", torch.tensor(cfg["mean"]))
            self.register_buffer("std", torch.tensor(cfg["std"]))

        self.activation = cfg["activation"]
        self.input_normalization = cfg["input_normalization"]
        if cfg["linear"]:
            self.linearl = nn.Linear(cfg["dim_input"], 16 * cfg["dim_input"])
        self.selfatt = nn.ModuleList()
        # dim_input = 16*dim_input
        self.selfatt1 = ISAB(
            16 * cfg["dim_input"],
            cfg["dim_hidden"],
            cfg["num_heads"],
            cfg["num_inds"],
            ln=cfg["ln"],
        )
        for i in range(cfg["n_l_enc"]):
            self.selfatt.append(
                ISAB(
                    cfg["dim_hidden"],
                    cfg["dim_hidden"],
                    cfg["num_heads"],
                    cfg["num_inds"],
                    ln=cfg["ln"],
                )
            )
        self.outatt = PMA(
            cfg["dim_hidden"], cfg["num_heads"], cfg["num_features"], ln=cfg["ln"]
        )

    def float2bit(
        self, f, num_e_bits=5, num_m_bits=10, bias=127.0, dtype=torch.float32
    ):
        """
        Convert floating-point numbers to a bit representation.

        Args:
            f (torch.Tensor): Input tensor of floating-point numbers.
            num_e_bits (int, optional): Number of bits for the exponent. Defaults to 5.
            num_m_bits (int, optional): Number of bits for the mantissa. Defaults to 10.
            bias (float, optional): Bias for the conversion. Defaults to 127.0.
            dtype (torch.dtype, optional): Data type for the output tensor. Defaults to torch.float32.

        Returns:
            torch.Tensor: Tensor with the bit representation of the input floating-point numbers.
        """
        ## SIGN BIT
        s = (
            torch.sign(f + 0.001) * -1 + 1
        ) * 0.5  # Swap plus and minus => 0 is plus and 1 is minus
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        ## EXPONENT BIT
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float("-inf")] = -(2 ** (num_e_bits - 1) - 1)
        e_decimal = e_scientific + (2 ** (num_e_bits - 1) - 1)
        e = self.integer2bit(e_decimal, num_bits=num_e_bits)
        ## MANTISSA
        f2 = f1 / 2 ** (e_scientific)
        m2 = self.remainder2bit(f2 % 1, num_bits=bias)
        fin_m = m2[:, :, :, :num_m_bits]  # [:,:,:,8:num_m_bits+8]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

    def remainder2bit(self, remainder, num_bits=127):
        """
        Convert remainder of floating-point number to bit representation.

        Args:
            remainder (torch.Tensor): Input tensor with remainders of floating-point numbers.
            num_bits (int, optional): Number of bits for the conversion. Defaults to 127.

        Returns:
            torch.Tensor: Tensor with the bit representation of the input remainders.
        """
        dtype = remainder.type()
        exponent_bits = torch.arange(num_bits, device=remainder.device).type(dtype)
        exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
        out = (remainder.unsqueeze(-1) * 2**exponent_bits) % 1
        return torch.floor(2 * out)

    def integer2bit(self, integer, num_bits=8):
        """
        Convert integer values to a bit representation.

        Args:
            integer (torch.Tensor): Input tensor of integer values.
            num_bits (int, optional): Number of bits for the conversion. Defaults to 8.

        Returns:
            torch.Tensor: Tensor with the bit representation of the input integer values.
        """
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1, device=integer.device).type(
            dtype
        )
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2**exponent_bits
        return (out - (out % 1)) % 2

    def forward(self, x):
        if self.bit16:
            x = self.float2bit(x)
            x = x.view(x.shape[0], x.shape[1], -1)
            if self.norm:
                x = (x - 0.5) * 2
        if self.input_normalization:
            means = x[:, :, -1].mean(axis=1).reshape(-1, 1)
            std = x[:, :, -1].std(axis=1).reshape(-1, 1)
            std[std == 0] = 1
            x[:, :, -1] = (x[:, :, -1] - means) / std

        if self.linear:
            if self.activation == "relu":
                x = torch.relu(self.linearl(x))
            elif self.activation == "sine":
                x = torch.sin(self.linearl(x))
            else:
                x = self.linearl(x)
        x = self.selfatt1(x)
        for layer in self.selfatt:
            x = layer(x)
        x = self.outatt(x)
        return x


class TreeEncoder(nn.Module):
    """
    Tree Encoder.

    Implements an encoder that takes tree-structured data and encodes it into a fixed-sized vector using
    Transformer architecture. The assumption is that the tree data has been flattened into a sequential
    representation that can be input to this model.

    Attributes:
        input_dim (int): The dimension of the input data.
        embed_dim (int): The desired embedding dimension after the initial linear layer.
        max_length (int): The maximum length of the sequential representation of the tree structure.
        embedding_layer (nn.Linear): Linear layer to transform the input data to the desired embedding dimension.
        transformer_encoder (nn.TransformerEncoder): The main encoder based on Transformer architecture.
        output_layer (nn.Linear): Linear layer to map from the embed_dim to the desired output_dim.
    """

    def __init__(self, cfg):
        """
        Initialize the TreeStructureEncoder.

        Args:
            input_dim (int): The dimension of the input data.
            max_length (int): The maximum length of the sequential representation of the tree structure.
            embed_dim (int): Desired embedding dimension.
            num_heads (int): Number of attention heads in the TransformerEncoder.
            num_encoder_layers (int): Number of layers in the TransformerEncoder.
            output_dim (int): Desired output dimension after the final linear layer.
        """
        super(TreeEncoder, self).__init__()

        self.max_length = cfg["dim_input"][0]
        self.embed_dim = cfg["dim_hidden"]
        self.embedding_layer = nn.Linear(cfg["dim_input"][1], cfg["dim_hidden"])
        self.pos_embedding = nn.Embedding(
            num_embeddings=cfg["dim_input"][1], embedding_dim=cfg["dim_hidden"]
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg["dim_hidden"], nhead=cfg["num_heads"]
            ),
            num_layers=cfg["num_layers"],
        )
        # self.transformer_encoder=nn.Sequential(*[Block(cfg) for _ in range(cfg["num_layers"])])

        self.output_layer = nn.Linear(cfg["dim_hidden"], cfg["dim_output"])

    def create_positional_encodings(self, max_length, embed_dim):
        """
        Create positional encodings.

        Args:
            max_length (int): Maximum length of the sequential representation.
            embed_dim (int): Embedding dimension.

        Returns:
            torch.Tensor: Tensor containing positional encodings of shape (max_length, embed_dim).
        """
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """
        Forward pass for TreeStructureEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_length, input_dim).

        Returns:
            torch.Tensor: Encoded output tensor after processing through the TransformerEncoder and
                          a final linear layer.
        """
        embeddings = self.embedding_layer(x)

        # Uncomment below if you want to use positional encodings
        positional_encodings = self.create_positional_encodings(
            self.max_length, self.embed_dim
        )
        positional_encodings = positional_encodings.repeat(x.shape[0], 1, 1).to(
            x.device
        )
        embeddings += positional_encodings

        embeddings = embeddings.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(embeddings)

        # Average over all sequence positions to get a fixed-sized representation
        sequence_representation = transformer_output.mean(dim=0)

        output = self.output_layer(sequence_representation)

        return output


# --- [Original file: SymQ.py] ---

class SymQ(nn.Module):
    """
    Attributes:
    - set_encoder (SetEncoder): Encoder for input point sets.
    - tree_encoder (TreeStructureEncoder): Encoder for tree-structured data.
    - device (torch.device): Device for storing tensors.
    """

    def __init__(self, cfg, device):
        super(SymQ, self).__init__()
        self.device = device
        self.set_encoder = SetEncoder(cfg["SetEncoder"]).to(device)
        self.tree_encoder = TreeEncoder(cfg["TreeEncoder"]).to(device)
        self.cfg = cfg["SymQ"]
        self.dropout = nn.Dropout(0)

        if self.cfg["batch_norm"]:
            self.batch_norm1 = nn.BatchNorm1d(self.cfg["dim_hidden"])
            self.batch_norm2 = nn.BatchNorm1d(self.cfg["dim_hidden"])

        if self.cfg["embedding_fusion"] == "concat":
            dim_fusion = (
                cfg["SetEncoder"]["dim_output"] + cfg["TreeEncoder"]["dim_output"]
            )
        elif self.cfg["embedding_fusion"] == "add":
            assert cfg["SetEncoder"]["dim_output"] == cfg["TreeEncoder"]["dim_output"]
            dim_fusion = cfg["SetEncoder"]["dim_output"]
        else:
            raise KeyError(
                f"Unknown embedding fusion method: {self.cfg['embedding_fusion']}"
            )
        self.downsample = nn.Linear(
            int(512 * cfg["SetEncoder"]["num_features"]),
            cfg["SetEncoder"]["dim_output"],
        )
        self.BN_downsample = nn.BatchNorm1d(cfg["SetEncoder"]["dim_output"])
        self.act_downsample = nn.GELU()
        self.linear1 = nn.Linear(dim_fusion, self.cfg["dim_hidden"])
        self.relu1 = nn.GELU()

        self.linear2 = nn.Linear(self.cfg["dim_hidden"], self.cfg["dim_hidden"])
        self.relu2 = nn.GELU()

        self.linear3 = nn.Linear(self.cfg["dim_hidden"], self.cfg["dim_hidden"])
        self.relu3 = nn.GELU()

        self.linear4 = nn.Linear(self.cfg["dim_hidden"], self.cfg["num_actions"])

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of the module.
        Uses Xavier Uniform initialization for linear layers and sets bias to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def _encode_set(self, x):
        x = x.permute(0, 2, 1)
        x = self.set_encoder(x)

        return x.view(x.size(0), -1)

    def _encode_tree(self, x):
        return self.tree_encoder(x)

    def forward(self, point_set, tree, support):
        # point_set: [batch_size, 2, num_point]
        # operator: [batch_size, seq_len, num_action]

        set_embedding = self._encode_set(point_set)
        set_embedding = self.act_downsample(
            self.BN_downsample(self.downsample(set_embedding))
        )
        forward_info = set_embedding.clone()
        tree_embedding = self._encode_tree(tree)

        if self.cfg["embedding_fusion"] == "concat":
            embeddings = torch.cat((self.dropout(forward_info), tree_embedding), 1)
        elif self.cfg["embedding_fusion"] == "add":
            embeddings = self.dropout(forward_info) + tree_embedding

        x = self.linear1(embeddings)
        if self.cfg["batch_norm"]:
            x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        if self.cfg["batch_norm"]:
            x = self.batch_norm2(x)
        fusion = x
        x = self.relu2(x)
        x = self.relu3(self.linear3(x))
        if self.cfg["set_skip_connection"]:
            assert x.shape == forward_info.shape
            x = x + forward_info
        x = self.linear4(x)
        q_values = x

        return q_values, set_embedding, set_embedding, fusion

    def act(self, point_set, tree):
        with torch.no_grad():
            q_values, _, _, _ = self.forward(point_set, tree, point_set)
        return q_values.argmax(dim=-1), q_values


# --- [Original file: loss.py] ---

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self, temperature=0.07, contrast_mode="all", base_temperature=0.07, device=None
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = self.device

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def _small_set_cfg(num_features=2):
    return {
        "linear": False,
        "bit16": True,
        "norm": True,
        "mean": 0.5,
        "std": 0.5,
        "activation": "relu",
        "input_normalization": False,
        "dim_input": 2,
        "dim_hidden": 512,
        "num_heads": 8,
        "num_inds": 2,
        "ln": True,
        "n_l_enc": 0,
        "num_features": num_features,
        "dim_output": 32,
    }


def _small_tree_cfg():
    return {
        "dim_input": [5, 6],
        "dim_hidden": 32,
        "num_heads": 4,
        "num_layers": 1,
        "dim_output": 32,
    }


def _small_symq_cfg(fusion="concat", skip=False):
    return {
        "SetEncoder": _small_set_cfg(num_features=2),
        "TreeEncoder": _small_tree_cfg(),
        "SymQ": {
            "embedding_fusion": fusion,
            "set_skip_connection": skip,
            "dim_hidden": 32,
            "batch_norm": False,
            "num_actions": 6,
        },
    }


# ============================================================
# __main__: Automated test suite for 6 ablated functions
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

    print("=" * 70)
    print("Sym-Q benchmark: core model components")
    print("=" * 70)

    # ==============================================================
    # Test 1/6: MAB.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/6] MAB.forward")
    try:
        mab = MAB(dim_Q=4, dim_K=4, dim_V=8, num_heads=2, ln=True)
        q = torch.randn(2, 3, 4, requires_grad=True)
        k = torch.randn(2, 5, 4, requires_grad=True)
        out = mab(q, k)
        check("MAB output not None", out is not None)
        if out is None:
            skip_checks(5, "MAB output missing")
        else:
            check("MAB output shape", tuple(out.shape) == (2, 3, 8), f"got {tuple(out.shape)}")
            check("MAB output finite", torch.isfinite(out).all().item())
            check("MAB layer norm centers features", out.mean(dim=-1).abs().max().item() < 1e-4)
            out.sum().backward()
            check("MAB query gradient", q.grad is not None and torch.isfinite(q.grad).all().item())
            check("MAB key gradient", k.grad is not None and torch.isfinite(k.grad).all().item())
    except Exception as exc:
        skip_checks(6, f"MAB.forward raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 2/6: SetEncoder.float2bit
    # ==============================================================
    print("-" * 60)
    print("[Test 2/6] SetEncoder.float2bit")
    try:
        set_encoder = SetEncoder(_small_set_cfg(num_features=2))
        values = torch.tensor([[[1.0, -2.0], [0.5, 0.0]]])
        bits = set_encoder.float2bit(values)
        check("float2bit output not None", bits is not None)
        if bits is None:
            skip_checks(4, "float2bit output missing")
        else:
            check("float2bit output shape", tuple(bits.shape) == (1, 2, 2, 16), f"got {tuple(bits.shape)}")
            check("float2bit finite", torch.isfinite(bits).all().item())
            check("float2bit sign convention", bits[0, 0, 0, 0].item() == 0.0 and bits[0, 0, 1, 0].item() == 1.0)
            binary_values = torch.logical_or(bits == 0, bits == 1).all().item()
            check("float2bit binary output", binary_values)
    except Exception as exc:
        skip_checks(5, f"SetEncoder.float2bit raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 3/6: SetEncoder.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/6] SetEncoder.forward")
    try:
        set_encoder = SetEncoder(_small_set_cfg(num_features=2))
        set_encoder.eval()
        point_set = torch.randn(2, 7, 2)
        out_a = set_encoder(point_set)
        out_b = set_encoder(point_set[:, torch.tensor([3, 0, 6, 1, 4, 2, 5]), :])
        check("SetEncoder output not None", out_a is not None)
        if out_a is None:
            skip_checks(5, "SetEncoder output missing")
        else:
            check("SetEncoder output shape", tuple(out_a.shape) == (2, 2, 512), f"got {tuple(out_a.shape)}")
            check("SetEncoder output finite", torch.isfinite(out_a).all().item())
            check("SetEncoder permutation invariant", torch.allclose(out_a, out_b, atol=1e-4))
            out_a.sum().backward()
            grad_ok = set_encoder.selfatt1.I.grad is not None and torch.isfinite(set_encoder.selfatt1.I.grad).all().item()
            check("SetEncoder inducing-point gradient", grad_ok)
            check("SetEncoder pooled seeds preserved", out_a.shape[1] == 2)
    except Exception as exc:
        skip_checks(6, f"SetEncoder.forward raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 4/6: TreeEncoder.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/6] TreeEncoder.forward")
    try:
        tree_encoder = TreeEncoder(_small_tree_cfg())
        tree_encoder.eval()
        tree = torch.zeros(2, 5, 6)
        tree[:, 0, 1] = 1.0
        tree[:, 2, 3] = 1.0
        tree[:, 4, 5] = 1.0
        shifted_tree = torch.roll(tree, shifts=1, dims=1)
        out = tree_encoder(tree)
        shifted_out = tree_encoder(shifted_tree)
        check("TreeEncoder output not None", out is not None)
        if out is None:
            skip_checks(4, "TreeEncoder output missing")
        else:
            check("TreeEncoder output shape", tuple(out.shape) == (2, 32), f"got {tuple(out.shape)}")
            check("TreeEncoder output finite", torch.isfinite(out).all().item())
            check("TreeEncoder positional signal affects output", not torch.allclose(out, shifted_out, atol=1e-5))
            out.sum().backward()
            grad_ok = tree_encoder.embedding_layer.weight.grad is not None and torch.isfinite(tree_encoder.embedding_layer.weight.grad).all().item()
            check("TreeEncoder embedding gradient", grad_ok)
    except Exception as exc:
        skip_checks(5, f"TreeEncoder.forward raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 5/6: SymQ.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 5/6] SymQ.forward")
    try:
        model = SymQ(_small_symq_cfg(fusion="concat", skip=False), torch.device("cpu"))
        point_set = torch.randn(2, 2, 7)
        tree = torch.zeros(2, 5, 6)
        tree[:, 1, 2] = 1.0
        tree[:, 3, 4] = 1.0
        q_values, set_embedding, support_embedding, fusion = model(point_set, tree, point_set)
        action, act_q_values = model.act(point_set, tree)
        check("SymQ q_values not None", q_values is not None)
        if q_values is None:
            skip_checks(6, "SymQ q_values missing")
        else:
            check("SymQ q_values shape", tuple(q_values.shape) == (2, 6), f"got {tuple(q_values.shape)}")
            check("SymQ set embedding shape", tuple(set_embedding.shape) == (2, 32), f"got {tuple(set_embedding.shape)}")
            check("SymQ fusion shape", tuple(fusion.shape) == (2, 32), f"got {tuple(fusion.shape)}")
            check("SymQ outputs finite", torch.isfinite(q_values).all().item() and torch.isfinite(fusion).all().item())
            check("SymQ support mirrors set embedding", torch.allclose(set_embedding, support_embedding))
            check("SymQ act uses argmax", torch.equal(action, act_q_values.argmax(dim=-1)))
    except Exception as exc:
        skip_checks(7, f"SymQ.forward raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 6/6: SupConLoss.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 6/6] SupConLoss.forward")
    try:
        loss_fn = SupConLoss(temperature=0.1, base_temperature=0.1, device=torch.device("cpu"))
        base = F.normalize(torch.randn(4, 8), p=2, dim=1)
        features = torch.stack([base, base + 0.01 * torch.randn_like(base)], dim=1).requires_grad_()
        labels = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(features, labels=labels)
        check("SupConLoss output not None", loss is not None)
        if loss is None:
            skip_checks(5, "SupConLoss output missing")
        else:
            check("SupConLoss scalar", tuple(loss.shape) == (), f"got {tuple(loss.shape)}")
            check("SupConLoss finite", torch.isfinite(loss).item())
            loss.backward()
            check("SupConLoss feature gradient", features.grad is not None and torch.isfinite(features.grad).all().item())
            mask_loss = loss_fn(features.detach(), mask=torch.eye(4))
            check("SupConLoss supports explicit mask", torch.isfinite(mask_loss).item())
            try:
                loss_fn(features.detach(), labels=labels, mask=torch.eye(4))
                raised = False
            except ValueError:
                raised = True
            check("SupConLoss rejects labels and mask together", raised)
    except Exception as exc:
        skip_checks(6, f"SupConLoss.forward raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
