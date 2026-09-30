"""Ground-truth core model components for the MuseMorphose benchmark.

This file consolidates the architecture-level Transformer VAE components used by
MuseMorphose: token/position embeddings, the VAE Transformer encoder, the
conditioned Transformer decoder, and the top-level MuseMorphose model.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


# --- [Original file: model/transformer_helpers.py] ---

def generate_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask


def weight_init_normal(weight, normal_std):
  nn.init.normal_(weight, 0.0, normal_std)


def weight_init_orthogonal(weight, gain):
  nn.init.orthogonal_(weight, gain)


def bias_init(bias):
  nn.init.constant_(bias, 0.0)
  

def weights_init(m):
    classname = m.__class__.__name__
    # print ('[{}] initializing ...'.format(classname))

    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            weight_init_normal(m.weight, 0.01)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('GRU') != -1:
        for param in m.parameters():
            if len(param.shape) >= 2:  # weights
                weight_init_orthogonal(param, 0.01)
            else:                      # biases
                bias_init(param)
    # else:
    #   print ('[{}] not initialized !!'.format(classname))


class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_pos=20480):
        super(PositionalEncoding, self).__init__()
        self.d_embed = d_embed
        self.max_pos = max_pos

        pe = torch.zeros(max_pos, d_embed)
        position = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, seq_len, bsz=None):
        pos_encoding = self.pe[:seq_len, :]

        if bsz is not None:
          pos_encoding = pos_encoding.expand(seq_len, bsz, -1)

        return pos_encoding


class TokenEmbedding(nn.Module):
  def __init__(self, n_token, d_embed, d_proj):
    super(TokenEmbedding, self).__init__()

    self.n_token = n_token
    self.d_embed = d_embed
    self.d_proj = d_proj
    self.emb_scale = d_proj ** 0.5

    self.emb_lookup = nn.Embedding(n_token, d_embed)
    if d_proj != d_embed:
      self.emb_proj = nn.Linear(d_embed, d_proj, bias=False)
    else:
      self.emb_proj = None

  def forward(self, inp_tokens):
    inp_emb = self.emb_lookup(inp_tokens)
    
    if self.emb_proj is not None:
      inp_emb = self.emb_proj(inp_emb)

    return inp_emb.mul_(self.emb_scale)


# --- [Original file: model/transformer_encoder.py] ---

class VAETransformerEncoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_vae_latent, dropout=0.1, activation='relu'):
    super(VAETransformerEncoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_vae_latent = d_vae_latent
    self.dropout = dropout
    self.activation = activation

    self.tr_encoder_layer = nn.TransformerEncoderLayer(
      d_model, n_head, d_ff, dropout, activation
    )
    self.tr_encoder = nn.TransformerEncoder(
      self.tr_encoder_layer, n_layer
    )

    self.fc_mu = nn.Linear(d_model, d_vae_latent)
    self.fc_logvar = nn.Linear(d_model, d_vae_latent)

  def forward(self, x, padding_mask=None):
    """
    TODO: Reproduce the VAE Transformer encoder forward pass.

    Input:
      x: (seq_len_per_bar, batch_times_bars, d_model) embedded bar-token sequence.
      padding_mask: optional (batch_times_bars, seq_len_per_bar) boolean mask where
        True marks padded positions.

    Output:
      hidden_out: (batch_times_bars, d_model) first-token encoder summary.
      mu: (batch_times_bars, d_vae_latent) latent mean.
      logvar: (batch_times_bars, d_vae_latent) latent log-variance.

"""
    pass


# --- [Original file: model/musemorphose.py] ---

class VAETransformerDecoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_seg_emb, dropout=0.1, activation='relu', cond_mode='in-attn'):
    super(VAETransformerDecoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_seg_emb = d_seg_emb
    self.dropout = dropout
    self.activation = activation
    self.cond_mode = cond_mode

    if cond_mode == 'in-attn':
      self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
    elif cond_mode == 'pre-attn':
      self.seg_emb_proj = nn.Linear(d_seg_emb + d_model, d_model, bias=False)

    self.decoder_layers = nn.ModuleList()
    for i in range(n_layer):
      self.decoder_layers.append(
        nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
      )

  def forward(self, x, seg_emb):
    """
    TODO: Reproduce the conditioned causal Transformer decoder.

    Input:
      x: (dec_seq_len, batch, d_model) decoder token embeddings.
      seg_emb: (dec_seq_len, batch, d_seg_emb) conditioning tensor containing
        bar-level VAE latent features and, when enabled, attribute embeddings.

    Output:
      (dec_seq_len, batch, d_model) decoded hidden states.

"""
    pass


class MuseMorphose(nn.Module):
  def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, 
    dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
    d_vae_latent, d_embed, n_token,
    enc_dropout=0.1, enc_activation='relu',
    dec_dropout=0.1, dec_activation='relu',
    d_rfreq_emb=32, d_polyph_emb=32,
    n_rfreq_cls=8, n_polyph_cls=8,
    is_training=True, use_attr_cls=True,
    cond_mode='in-attn'
  ):
    super(MuseMorphose, self).__init__()
    self.enc_n_layer = enc_n_layer
    self.enc_n_head = enc_n_head
    self.enc_d_model = enc_d_model
    self.enc_d_ff = enc_d_ff
    self.enc_dropout = enc_dropout
    self.enc_activation = enc_activation

    self.dec_n_layer = dec_n_layer
    self.dec_n_head = dec_n_head
    self.dec_d_model = dec_d_model
    self.dec_d_ff = dec_d_ff
    self.dec_dropout = dec_dropout
    self.dec_activation = dec_activation  

    self.d_vae_latent = d_vae_latent
    self.n_token = n_token
    self.is_training = is_training

    self.cond_mode = cond_mode
    self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
    self.d_embed = d_embed
    self.pe = PositionalEncoding(d_embed)
    self.dec_out_proj = nn.Linear(dec_d_model, n_token)
    self.encoder = VAETransformerEncoder(
      enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_vae_latent, enc_dropout, enc_activation
    )

    self.use_attr_cls = use_attr_cls
    if use_attr_cls:
      self.decoder = VAETransformerDecoder(
        dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_polyph_emb + d_rfreq_emb,
        dropout=dec_dropout, activation=dec_activation,
        cond_mode=cond_mode
      )
    else:
      self.decoder = VAETransformerDecoder(
        dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent,
        dropout=dec_dropout, activation=dec_activation,
        cond_mode=cond_mode
      )

    if use_attr_cls:
      self.d_rfreq_emb = d_rfreq_emb
      self.d_polyph_emb = d_polyph_emb
      self.rfreq_attr_emb = TokenEmbedding(n_rfreq_cls, d_rfreq_emb, d_rfreq_emb)
      self.polyph_attr_emb = TokenEmbedding(n_polyph_cls, d_polyph_emb, d_polyph_emb)
    else:
      self.rfreq_attr_emb = None
      self.polyph_attr_emb = None

    self.emb_dropout = nn.Dropout(self.enc_dropout)
    self.apply(weights_init)
    

  def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.):
    """
    TODO: Reproduce the VAE latent reparameterization step.

    Input:
      mu: (..., d_vae_latent) latent mean tensor.
      logvar: (..., d_vae_latent) latent log-variance tensor.
      use_sampling: whether to draw stochastic noise or use deterministic zero noise.
      sampling_var: scalar multiplier controlling sampling noise scale.

    Output:
      (..., d_vae_latent) latent tensor on the same device as mu.

"""
    pass

  def get_sampled_latent(self, inp, padding_mask=None, use_sampling=False, sampling_var=0.):
    token_emb = self.token_emb(inp)
    enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

    _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
    mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
    vae_latent = self.reparameterize(mu, logvar, use_sampling=use_sampling, sampling_var=sampling_var)

    return vae_latent

  def generate(self, inp, dec_seg_emb, rfreq_cls=None, polyph_cls=None, keep_last_only=True):
    """
    TODO: Reproduce MuseMorphose autoregressive generation scoring.

    Input:
      inp: (dec_seq_len, batch) decoder token ids.
      dec_seg_emb: (dec_seq_len, batch, d_vae_latent) per-timestep latent condition.
      rfreq_cls: optional (dec_seq_len, batch) rhythmic-frequency class ids.
      polyph_cls: optional (dec_seq_len, batch) polyphonicity class ids.
      keep_last_only: whether to return only the final timestep logits.

    Output:
      If keep_last_only is True: (batch, n_token) logits.
      Otherwise: (dec_seq_len, batch, n_token) logits.

"""
    pass


  def forward(self, enc_inp, dec_inp, dec_inp_bar_pos, rfreq_cls=None, polyph_cls=None, padding_mask=None):
    """
    TODO: Reproduce the full MuseMorphose Transformer-VAE forward pass.

    Input:
      enc_inp: (enc_seq_len_per_bar, batch, n_bars) encoder token ids.
      dec_inp: (dec_seq_len_per_sample, batch) decoder token ids.
      dec_inp_bar_pos: (batch, n_bars + 1) start/end positions mapping bars to
        decoder timesteps.
      rfreq_cls: optional (dec_seq_len_per_sample, batch) rhythmic attribute ids.
      polyph_cls: optional (dec_seq_len_per_sample, batch) polyphonic attribute ids.
      padding_mask: optional (batch, n_bars, enc_seq_len_per_bar) encoder padding mask.

    Output:
      mu: (batch * n_bars, d_vae_latent) latent means.
      logvar: (batch * n_bars, d_vae_latent) latent log-variances.
      dec_logits: (dec_seq_len_per_sample, batch, n_token) decoder token logits.

"""
    pass

  def compute_loss(self, mu, logvar, beta, fb_lambda, dec_logits, dec_tgt):
    recons_loss = F.cross_entropy(
      dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
      ignore_index=self.n_token - 1, reduction='mean'
    ).float()

    kl_raw = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
    kl_before_free_bits = kl_raw.mean()
    kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
    kldiv_loss = kl_after_free_bits.mean()

    return {
      'beta': beta,
      'total_loss': recons_loss + beta * kldiv_loss,
      'kldiv_loss': kldiv_loss,
      'kldiv_raw': kl_before_free_bits,
      'recons_loss': recons_loss
    }


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

    def make_small_model(use_attr_cls=True, cond_mode="in-attn"):
        return MuseMorphose(
            enc_n_layer=1,
            enc_n_head=2,
            enc_d_model=8,
            enc_d_ff=16,
            dec_n_layer=1,
            dec_n_head=2,
            dec_d_model=8,
            dec_d_ff=16,
            d_vae_latent=4,
            d_embed=8,
            n_token=20,
            enc_dropout=0.0,
            dec_dropout=0.0,
            d_rfreq_emb=2,
            d_polyph_emb=2,
            n_rfreq_cls=4,
            n_polyph_cls=4,
            use_attr_cls=use_attr_cls,
            cond_mode=cond_mode,
        )

    print("[Test 1/5] VAETransformerEncoder.forward")
    try:
        encoder = VAETransformerEncoder(1, 2, 8, 16, 4, dropout=0.0)
        x = torch.randn(5, 3, 8, requires_grad=True)
        padding_mask = torch.zeros(3, 5, dtype=torch.bool)
        hidden, mu, logvar = encoder(x, padding_mask=padding_mask)
        check("Encoder output not None", hidden is not None and mu is not None and logvar is not None)
        if hidden is not None and mu is not None and logvar is not None:
            check("Encoder hidden shape", tuple(hidden.shape) == (3, 8), str(tuple(hidden.shape)))
            check("Encoder latent shapes", tuple(mu.shape) == (3, 4) and tuple(logvar.shape) == (3, 4))
            check("Encoder finite outputs", torch.isfinite(hidden).all().item() and torch.isfinite(mu).all().item() and torch.isfinite(logvar).all().item())
            (hidden.sum() + mu.sum() + logvar.sum()).backward()
            check("Encoder gradient reaches input", x.grad is not None and x.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "VAETransformerEncoder returned None")
    except Exception as exc:
        skip_checks(5, f"VAETransformerEncoder.forward raised {type(exc).__name__}: {exc}")

    print("[Test 2/5] VAETransformerDecoder.forward")
    try:
        decoder = VAETransformerDecoder(1, 2, 8, 16, 4, dropout=0.0, cond_mode="in-attn")
        x_leaf = torch.randn(6, 3, 8, requires_grad=True)
        x = x_leaf * 1.0
        seg_emb = torch.randn(6, 3, 4, requires_grad=True)
        out = decoder(x, seg_emb)
        pre_decoder = VAETransformerDecoder(1, 2, 8, 16, 4, dropout=0.0, cond_mode="pre-attn")
        pre_out = pre_decoder(torch.randn(6, 3, 8), torch.randn(6, 3, 4))
        check("Decoder output not None", out is not None)
        if out is not None:
            check("Decoder output shape", tuple(out.shape) == (6, 3, 8), str(tuple(out.shape)))
            check("Decoder output finite", torch.isfinite(out).all().item())
            check("Decoder pre-attn branch shape", tuple(pre_out.shape) == (6, 3, 8), str(tuple(pre_out.shape)))
            out.sum().backward()
            check("Decoder gradients reach inputs", x_leaf.grad is not None and x_leaf.grad.abs().sum().item() > 0 and seg_emb.grad is not None and seg_emb.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "VAETransformerDecoder returned None")
    except Exception as exc:
        skip_checks(5, f"VAETransformerDecoder.forward raised {type(exc).__name__}: {exc}")

    print("[Test 3/5] MuseMorphose.reparameterize")
    try:
        model = make_small_model()
        mu = torch.randn(3, 4, requires_grad=True)
        logvar = torch.randn(3, 4, requires_grad=True)
        deterministic = model.reparameterize(mu, logvar, use_sampling=False)
        sampled = model.reparameterize(mu, logvar, use_sampling=True, sampling_var=1.0)
        check("Reparameterize output not None", deterministic is not None and sampled is not None)
        if deterministic is not None and sampled is not None:
            check("Reparameterize output shape", tuple(sampled.shape) == (3, 4), str(tuple(sampled.shape)))
            check("Reparameterize deterministic path equals mu", torch.allclose(deterministic, mu, atol=1e-6))
            check("Reparameterize finite sampled output", torch.isfinite(sampled).all().item())
            sampled.sum().backward()
            check("Reparameterize gradients reach mu/logvar", mu.grad is not None and mu.grad.abs().sum().item() > 0 and logvar.grad is not None and logvar.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "MuseMorphose.reparameterize returned None")
    except Exception as exc:
        skip_checks(5, f"MuseMorphose.reparameterize raised {type(exc).__name__}: {exc}")

    print("[Test 4/5] MuseMorphose.generate")
    try:
        model = make_small_model(use_attr_cls=True)
        model.eval()
        inp = torch.randint(0, 19, (5, 2))
        dec_seg_emb = torch.randn(5, 2, 4)
        rfreq_cls = torch.randint(0, 4, (5, 2))
        polyph_cls = torch.randint(0, 4, (5, 2))
        with torch.no_grad():
            out_last = model.generate(inp, dec_seg_emb, rfreq_cls=rfreq_cls, polyph_cls=polyph_cls, keep_last_only=True)
            out_all = model.generate(inp, dec_seg_emb, rfreq_cls=rfreq_cls, polyph_cls=polyph_cls, keep_last_only=False)
        check("Generate output not None", out_last is not None and out_all is not None)
        if out_last is not None and out_all is not None:
            check("Generate last-token shape", tuple(out_last.shape) == (2, 20), str(tuple(out_last.shape)))
            check("Generate full-sequence shape", tuple(out_all.shape) == (5, 2, 20), str(tuple(out_all.shape)))
            check("Generate finite logits", torch.isfinite(out_last).all().item() and torch.isfinite(out_all).all().item())
            check("Generate last-token matches final timestep", torch.allclose(out_last, out_all[-1], atol=1e-6))
        else:
            skip_checks(4, "MuseMorphose.generate returned None")
    except Exception as exc:
        skip_checks(5, f"MuseMorphose.generate raised {type(exc).__name__}: {exc}")

    print("[Test 5/5] MuseMorphose.forward")
    try:
        model = make_small_model(use_attr_cls=True)
        model.eval()
        enc_inp = torch.randint(0, 19, (4, 2, 3))
        dec_inp = torch.randint(0, 19, (7, 2))
        dec_inp_bar_pos = torch.tensor([[0, 2, 5, 7], [0, 3, 5, 7]], dtype=torch.long)
        rfreq_cls = torch.randint(0, 4, (7, 2))
        polyph_cls = torch.randint(0, 4, (7, 2))
        padding_mask = torch.zeros(2, 3, 4, dtype=torch.bool)
        mu, logvar, dec_logits = model(
            enc_inp,
            dec_inp,
            dec_inp_bar_pos,
            rfreq_cls=rfreq_cls,
            polyph_cls=polyph_cls,
            padding_mask=padding_mask,
        )
        check("MuseMorphose forward output not None", mu is not None and logvar is not None and dec_logits is not None)
        if mu is not None and logvar is not None and dec_logits is not None:
            check("MuseMorphose latent shapes", tuple(mu.shape) == (6, 4) and tuple(logvar.shape) == (6, 4))
            check("MuseMorphose logits shape", tuple(dec_logits.shape) == (7, 2, 20), str(tuple(dec_logits.shape)))
            check("MuseMorphose finite outputs", torch.isfinite(mu).all().item() and torch.isfinite(logvar).all().item() and torch.isfinite(dec_logits).all().item())
            check("MuseMorphose bar count flattened", mu.size(0) == enc_inp.size(1) * enc_inp.size(2), str(mu.size(0)))
        else:
            skip_checks(4, "MuseMorphose.forward returned None")
    except Exception as exc:
        skip_checks(5, f"MuseMorphose.forward raised {type(exc).__name__}: {exc}")

    print(f"RESULT: passed={passed} failed={failed}")
    if failed != 0:
        raise SystemExit(1)
