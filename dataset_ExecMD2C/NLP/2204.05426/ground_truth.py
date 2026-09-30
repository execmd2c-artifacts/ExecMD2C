from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TinyEncoderOutput:
    last_hidden_state: torch.Tensor


@dataclass
class TinySeq2SeqOutput:
    loss: torch.Tensor
    encoder_last_hidden_state: torch.Tensor


@dataclass
class TinyBartConfig:
    d_model: int = 8
    max_position_embeddings: int = 256
    pad_token_id: int = 1


class TinyBartEncoder(nn.Module):
    def __init__(self, config: TinyBartConfig, vocab_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layers = nn.ModuleList([nn.Linear(config.d_model, config.d_model) for _ in range(num_layers)])

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        del output_attentions, output_hidden_states
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        hidden = self.embed_tokens(input_ids) + self.embed_positions(positions)
        hidden = self.layernorm_embedding(hidden)
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        if attention_mask is not None:
            hidden = hidden * attention_mask.to(hidden.dtype).unsqueeze(-1)
        return TinyEncoderOutput(last_hidden_state=hidden)


class TinyBartForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: Optional[TinyBartConfig] = None,
        vocab_size: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.config = config or TinyBartConfig()
        self.vocab_size = vocab_size
        self.base_model = nn.Module()
        self.base_model.encoder = TinyBartEncoder(self.config, vocab_size=vocab_size, num_layers=num_layers)
        self.base_model.decoder = nn.Module()
        self.base_model.decoder.layers = nn.ModuleList(
            [nn.Linear(self.config.d_model, self.config.d_model) for _ in range(num_layers)]
        )
        self.model = nn.Module()
        self.model.shared = self.base_model.encoder.embed_tokens
        self.lm_head = nn.Linear(self.config.d_model, vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        encoder_output = self.base_model.encoder(
            input_ids,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if labels is None:
            loss = encoder_output.last_hidden_state.new_tensor(0.0)
        else:
            targets = labels.clone()
            valid = targets.ne(-100)
            targets[valid] = targets[valid].remainder(self.vocab_size)
            if valid.any():
                logits = self.lm_head(encoder_output.last_hidden_state)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="mean",
                )
            else:
                loss = encoder_output.last_hidden_state.new_tensor(0.0)
        return TinySeq2SeqOutput(loss=loss, encoder_last_hidden_state=encoder_output.last_hidden_state)


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _set_last_layer_status(layers: nn.ModuleList, status: bool):
    for index, layer in enumerate(layers):
        layer.requires_grad_(index == len(layers) - 1 and status)


def prototype_distance_matrix(last_hidden_state: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    if last_hidden_state.dim() != 3:
        raise ValueError("last_hidden_state must have shape [batch, sequence, hidden]")
    if prototypes.dim() != 3:
        raise ValueError("prototypes must have shape [num_prototypes, sequence, hidden]")
    if last_hidden_state.shape[1:] != prototypes.shape[1:]:
        raise ValueError("hidden states and prototypes must share sequence and hidden dimensions")
    batch_size = last_hidden_state.size(0)
    num_prototypes = prototypes.size(0)
    return torch.cdist(last_hidden_state.reshape(batch_size, -1), prototypes.reshape(num_prototypes, -1))


def class_aware_distance_mask(
    labels: torch.Tensor,
    num_prototypes: int,
    num_pos_prototypes: int,
    device: Optional[torch.device] = None,
    high_value: float = 1e7,
) -> torch.Tensor:
    if num_pos_prototypes < 0 or num_pos_prototypes > num_prototypes:
        raise ValueError("num_pos_prototypes must be between 0 and num_prototypes")
    label_device = device or labels.device
    distance_grounder = torch.zeros(2, num_prototypes, device=label_device)
    distance_grounder[0, :num_pos_prototypes] = high_value
    distance_grounder[1, num_pos_prototypes:] = high_value
    return distance_grounder[labels.to(label_device).long()]


class SimpleBartModel(nn.Module):
    def __init__(self, n_classes=2, hidden_size: int = 8, vocab_size: int = 64):
        super().__init__()
        config = TinyBartConfig(d_model=hidden_size)
        self.bart_encoder_model = TinyBartEncoder(config, vocab_size=vocab_size)
        self.num_enc_layers = len(self.bart_encoder_model.layers)
        _set_last_layer_status(self.bart_encoder_model.layers, True)
        self.bart_out_dim = self.bart_encoder_model.config.d_model
        self.classfn_model = nn.Linear(self.bart_out_dim, n_classes)
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, input_ids, attn_mask, y, use_decoder=0, use_classfn=1):
        del use_decoder, use_classfn
        device = _module_device(self)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        eos_mask = input_ids.eq(2)
        last_hidden_state = self.bart_encoder_model(
            input_ids,
            attn_mask,
            output_attentions=False,
            output_hidden_states=False,
        ).last_hidden_state
        sentence_representation = last_hidden_state[eos_mask, :].view(
            last_hidden_state.size(0),
            -1,
            last_hidden_state.size(-1),
        )[:, -1, :]
        classfn_out = self.classfn_model(sentence_representation)
        classfn_loss = self.loss_fn(classfn_out, y.to(device))
        return classfn_out, classfn_loss


class SimpleProtoTex(nn.Module):
    def __init__(
        self,
        num_prototypes=20,
        n_classes=2,
        hidden_size: int = 8,
        max_position_embeddings: int = 256,
        vocab_size: int = 64,
    ):
        super().__init__()
        config = TinyBartConfig(d_model=hidden_size, max_position_embeddings=max_position_embeddings)
        self.bart_model = TinyBartForConditionalGeneration(config, vocab_size=vocab_size)
        self.bart_out_dim = self.bart_model.config.d_model
        self.max_position_embeddings = max_position_embeddings
        self.num_protos = num_prototypes
        self.prototypes = nn.Parameter(torch.rand(self.num_protos, self.max_position_embeddings, self.bart_out_dim))
        self.classfn_model = nn.Linear(self.num_protos, n_classes)
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.set_encoder_status(True)
        self.set_decoder_status(False)
        self.set_protos_status(False)
        self.set_classfn_status(False)
        self.BNLayer = nn.BatchNorm1d(self.num_protos)

    def set_encoder_status(self, status=True):
        self.num_enc_layers = len(self.bart_model.base_model.encoder.layers)
        _set_last_layer_status(self.bart_model.base_model.encoder.layers, status)

    def set_decoder_status(self, status=True):
        self.num_dec_layers = len(self.bart_model.base_model.decoder.layers)
        _set_last_layer_status(self.bart_model.base_model.decoder.layers, status)

    def set_classfn_status(self, status=True):
        self.classfn_model.requires_grad_(status)

    def set_protos_status(self, status=True):
        self.prototypes.requires_grad = status

    def forward(
        self,
        input_ids,
        attn_mask,
        y,
        use_decoder=1,
        use_classfn=0,
        use_rc=0,
        use_p1=0,
        use_p2=0,
        rc_loss_lamb=0.95,
        p1_lamb=0.93,
        p2_lamb=0.92,
    ):
        device = _module_device(self)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        y = y.to(device)
        batch_size = input_ids.size(0)
        if use_decoder:
            labels = input_ids.clone()
            labels[labels == self.bart_model.config.pad_token_id] = -100
            bart_output = self.bart_model(
                input_ids,
                attn_mask,
                labels=labels,
                output_attentions=False,
                output_hidden_states=False,
            )
            rc_loss = batch_size * bart_output.loss
            last_hidden_state = bart_output.encoder_last_hidden_state
        else:
            rc_loss = input_ids.new_tensor(0.0, dtype=torch.float32)
            last_hidden_state = self.bart_model.base_model.encoder(
                input_ids,
                attn_mask,
                output_attentions=False,
                output_hidden_states=False,
            ).last_hidden_state

        input_for_classfn, l_p1, l_p2, classfn_out, classfn_loss = (
            None,
            input_ids.new_tensor(0.0, dtype=torch.float32),
            input_ids.new_tensor(0.0, dtype=torch.float32),
            None,
            input_ids.new_tensor(0.0, dtype=torch.float32),
        )
        if use_classfn or use_p1 or use_p2:
            input_for_classfn = prototype_distance_matrix(last_hidden_state, self.prototypes)
        if use_p1:
            l_p1 = torch.mean(torch.min(input_for_classfn, dim=0)[0])
        if use_p2:
            l_p2 = torch.mean(torch.min(input_for_classfn, dim=1)[0])
        if use_classfn:
            classfn_out = self.classfn_model(input_for_classfn).view(batch_size, -1)
            classfn_loss = self.loss_fn(classfn_out, y)
        if not use_rc:
            rc_loss = input_ids.new_tensor(0.0, dtype=torch.float32)
        total_loss = classfn_loss + rc_loss_lamb * rc_loss + p1_lamb * l_p1 + p2_lamb * l_p2
        return classfn_out, (
            total_loss,
            classfn_loss.detach().cpu(),
            rc_loss.detach().cpu() if torch.is_tensor(rc_loss) else rc_loss,
            l_p1.detach().cpu(),
            l_p2.detach().cpu(),
        )


class ProtoTEx(nn.Module):
    def __init__(
        self,
        num_prototypes,
        num_pos_prototypes,
        n_classes=2,
        bias=True,
        dropout=False,
        special_classfn=False,
        p=0.5,
        batchnormlp1=False,
        hidden_size: int = 8,
        max_position_embeddings: int = 256,
        vocab_size: int = 64,
    ):
        super().__init__()
        if num_pos_prototypes < 0 or num_pos_prototypes > num_prototypes:
            raise ValueError("num_pos_prototypes must be between 0 and num_prototypes")
        config = TinyBartConfig(d_model=hidden_size, max_position_embeddings=max_position_embeddings)
        self.bart_model = TinyBartForConditionalGeneration(config, vocab_size=vocab_size)
        self.bart_out_dim = self.bart_model.config.d_model
        self.one_by_sqrt_bartoutdim = 1 / torch.sqrt(torch.tensor(self.bart_out_dim).float())
        self.max_position_embeddings = max_position_embeddings
        self.num_protos = num_prototypes
        self.num_pos_protos = num_pos_prototypes
        self.num_neg_protos = self.num_protos - self.num_pos_protos
        self.pos_prototypes = nn.Parameter(torch.rand(self.num_pos_protos, self.max_position_embeddings, self.bart_out_dim))
        self.neg_prototypes = nn.Parameter(torch.rand(self.num_neg_protos, self.max_position_embeddings, self.bart_out_dim))
        self.classfn_model = nn.Linear(self.num_protos, n_classes, bias=bias)
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.do_dropout = dropout
        self.special_classfn = special_classfn
        self.dropout = nn.Dropout(p=p)
        self.dobatchnorm = batchnormlp1
        distance_grounder = torch.zeros(2, self.num_protos)
        distance_grounder[0, : self.num_pos_protos] = 1e7
        distance_grounder[1, self.num_pos_protos :] = 1e7
        self.register_buffer("distance_grounder", distance_grounder)

    def set_prototypes(self, do_random=False):
        if do_random:
            nn.init.xavier_normal_(self.pos_prototypes)
            nn.init.xavier_normal_(self.neg_prototypes)
        else:
            raise ValueError("Encoded prototype initialization requires external selected examples")

    def set_shared_status(self, status=True):
        self.bart_model.model.shared.requires_grad_(status)

    def set_encoder_status(self, status=True):
        self.num_enc_layers = len(self.bart_model.base_model.encoder.layers)
        _set_last_layer_status(self.bart_model.base_model.encoder.layers, status)

    def set_decoder_status(self, status=True):
        self.num_dec_layers = len(self.bart_model.base_model.decoder.layers)
        _set_last_layer_status(self.bart_model.base_model.decoder.layers, status)

    def set_classfn_status(self, status=True):
        self.classfn_model.requires_grad_(status)

    def set_protos_status(self, pos_or_neg=None, status=True):
        if pos_or_neg == "pos" or pos_or_neg is None:
            self.pos_prototypes.requires_grad = status
        if pos_or_neg == "neg" or pos_or_neg is None:
            self.neg_prototypes.requires_grad = status

    def forward(
        self,
        input_ids,
        attn_mask,
        y,
        use_decoder=1,
        use_classfn=0,
        use_rc=0,
        use_p1=0,
        use_p2=0,
        use_p3=0,
        classfn_lamb=1.0,
        rc_loss_lamb=0.95,
        p1_lamb=0.93,
        p2_lamb=0.92,
        p3_lamb=1.0,
        distmask_lp1=0,
        distmask_lp2=0,
        pos_or_neg=None,
        random_mask_for_distanceMat=None,
    ):
        del pos_or_neg
        device = _module_device(self)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        y = y.to(device)
        batch_size = input_ids.size(0)
        if use_decoder:
            labels = input_ids.clone()
            labels[labels == self.bart_model.config.pad_token_id] = -100
            bart_output = self.bart_model(
                input_ids,
                attn_mask,
                labels=labels,
                output_attentions=False,
                output_hidden_states=False,
            )
            rc_loss = bart_output.loss
            last_hidden_state = bart_output.encoder_last_hidden_state
        else:
            rc_loss = input_ids.new_tensor(0.0, dtype=torch.float32)
            last_hidden_state = self.bart_model.base_model.encoder(
                input_ids,
                attn_mask,
                output_attentions=False,
                output_hidden_states=False,
            ).last_hidden_state

        input_for_classfn, l_p1, l_p2, l_p3, classfn_out, classfn_loss = (
            None,
            input_ids.new_tensor(0.0, dtype=torch.float32),
            input_ids.new_tensor(0.0, dtype=torch.float32),
            input_ids.new_tensor(0.0, dtype=torch.float32),
            None,
            input_ids.new_tensor(0.0, dtype=torch.float32),
        )
        if use_classfn or use_p1 or use_p2 or use_p3:
            all_protos = torch.cat((self.pos_prototypes, self.neg_prototypes), dim=0)
            if use_classfn or use_p1 or use_p2:
                input_for_classfn = prototype_distance_matrix(last_hidden_state, all_protos)
                if self.dobatchnorm:
                    input_for_classfn = F.instance_norm(
                        input_for_classfn.view(batch_size, 1, self.num_protos)
                    ).view(batch_size, self.num_protos)
            if use_p1 or use_p2:
                distance_mask = self.distance_grounder[y.long()]
                input_for_classfn_masked = input_for_classfn + distance_mask
                if random_mask_for_distanceMat:
                    random_mask = torch.bernoulli(
                        torch.ones_like(input_for_classfn_masked) * random_mask_for_distanceMat
                    ).bool()
                    input_for_classfn_masked = input_for_classfn_masked.masked_fill(random_mask, 1e7)
        if use_p1:
            l_p1 = torch.mean(torch.min(input_for_classfn_masked if distmask_lp1 else input_for_classfn, dim=0)[0])
        if use_p2:
            l_p2 = torch.mean(torch.min(input_for_classfn_masked if distmask_lp2 else input_for_classfn, dim=1)[0])
        if use_p3 and self.num_pos_protos > 1:
            l_p3 = self.one_by_sqrt_bartoutdim.to(device) * torch.mean(
                torch.pdist(self.pos_prototypes.view(self.num_pos_protos, -1))
            )
        if use_classfn:
            if self.do_dropout:
                if self.special_classfn:
                    classfn_out = (
                        input_for_classfn @ self.classfn_model.weight.t()
                        + self.dropout(self.classfn_model.bias.repeat(batch_size, 1))
                    ).view(batch_size, -1)
                else:
                    classfn_out = self.classfn_model(self.dropout(input_for_classfn)).view(batch_size, -1)
            else:
                classfn_out = self.classfn_model(input_for_classfn).view(batch_size, -1)
            classfn_loss = self.loss_fn(classfn_out, y)
        if not use_rc:
            rc_loss = input_ids.new_tensor(0.0, dtype=torch.float32)
        total_loss = classfn_lamb * classfn_loss + rc_loss_lamb * rc_loss + p1_lamb * l_p1 + p2_lamb * l_p2 - p3_lamb * l_p3
        return classfn_out, (
            total_loss,
            classfn_loss.detach().cpu(),
            rc_loss.detach().cpu(),
            l_p1.detach().cpu(),
            l_p2.detach().cpu(),
            l_p3.detach().cpu(),
        )


SimpleProtoTEx = SimpleProtoTex


if __name__ == "__main__":
    torch.manual_seed(7)
    passed = 0
    failed = 0

    def check(name, condition, details=""):
        global passed, failed
        if bool(condition):
            passed += 1
            print(f"PASS: {name}")
        else:
            failed += 1
            print(f"FAIL: {name} {details}")

    def skip_checks(count, reason):
        global failed
        failed += count
        print(f"SKIP: {count} checks marked failed because {reason}")

    def close(a, b, atol=1e-5):
        return torch.allclose(a, b, atol=atol, rtol=1e-5)

    class StaticEncoder(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = nn.Parameter(hidden.clone(), requires_grad=False)
            self.layers = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])

        def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False):
            del attention_mask, output_attentions, output_hidden_states
            return TinyEncoderOutput(last_hidden_state=self.hidden.to(input_ids.device))

    hidden = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[2.0, 0.0], [2.0, 2.0]],
        ]
    )
    protos = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[2.0, 0.0], [2.0, 1.0]],
            [[10.0, 0.0], [10.0, 0.0]],
        ]
    )
    expected_distances = torch.tensor(
        [
            [0.0, torch.sqrt(torch.tensor(6.0)), torch.sqrt(torch.tensor(181.0))],
            [3.0, 1.0, torch.sqrt(torch.tensor(132.0))],
        ]
    )
    try:
        distances = prototype_distance_matrix(hidden, protos)
        if distances is None:
            skip_checks(3, "prototype_distance_matrix returned None")
        else:
            check("distance matrix has batch-by-prototype shape", distances.shape == (2, 3), str(distances.shape))
            check("distance matrix equals flattened Euclidean distances", close(distances, expected_distances), distances)
            check("nearest prototypes are recovered from distances", torch.equal(distances.argmin(dim=1), torch.tensor([0, 1])))
    except Exception as exc:
        skip_checks(3, f"prototype distance checks raised {type(exc).__name__}: {exc}")

    labels = torch.tensor([0, 1])
    expected_mask = torch.tensor(
        [
            [1e7, 1e7, 1e7, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1e7, 1e7],
        ]
    )
    try:
        mask = class_aware_distance_mask(labels, num_prototypes=5, num_pos_prototypes=3)
        if mask is None:
            skip_checks(3, "class_aware_distance_mask returned None")
        else:
            check("class-aware mask has one row per label", mask.shape == (2, 5), str(mask.shape))
            check("class-aware mask blocks opposite prototype partition", close(mask, expected_mask), mask)
            masked_probe = torch.tensor([[0.1, 0.2, 0.3, 5.0, 6.0], [7.0, 6.0, 5.0, 0.1, 0.2]]) + mask
            check("class-aware mask redirects nearest choices", torch.equal(masked_probe.argmin(dim=1), torch.tensor([3, 2])))
    except Exception as exc:
        skip_checks(3, f"class-aware mask checks raised {type(exc).__name__}: {exc}")

    input_ids = torch.tensor([[3, 4], [5, 6]])
    attn_mask = torch.ones_like(input_ids)
    try:
        simple = SimpleProtoTex(num_prototypes=3, n_classes=2, hidden_size=2, max_position_embeddings=2, vocab_size=16)
        simple.bart_model.base_model.encoder = StaticEncoder(hidden)
        with torch.no_grad():
            simple.prototypes.copy_(protos)
            simple.classfn_model.weight.copy_(torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
            simple.classfn_model.bias.zero_()
        simple_result = simple(
            input_ids,
            attn_mask,
            labels,
            use_decoder=0,
            use_classfn=1,
            use_p1=1,
            use_p2=1,
        )
        if not isinstance(simple_result, tuple) or len(simple_result) != 2:
            skip_checks(3, "SimpleProtoTex did not return logits and losses")
        else:
            simple_logits, simple_losses = simple_result
            expected_simple_logits = simple.classfn_model(expected_distances)
            expected_simple_lp1 = torch.mean(torch.min(expected_distances, dim=0)[0])
            expected_simple_lp2 = torch.mean(torch.min(expected_distances, dim=1)[0])
            expected_simple_ce = F.cross_entropy(expected_simple_logits, labels)
            expected_simple_total = expected_simple_ce + 0.93 * expected_simple_lp1 + 0.92 * expected_simple_lp2
            simple_losses_ok = isinstance(simple_losses, tuple) and len(simple_losses) > 4
            check("SimpleProtoTex classification logits come from prototype distances", close(simple_logits, expected_simple_logits), simple_logits)
            check("SimpleProtoTex lp1/lp2 match nearest prototype losses", simple_losses_ok and close(simple_losses[3], expected_simple_lp1) and close(simple_losses[4], expected_simple_lp2))
            check("SimpleProtoTex total loss combines class, lp1, and lp2 terms", simple_losses_ok and close(simple_losses[0], expected_simple_total), simple_losses)
    except Exception as exc:
        skip_checks(3, f"SimpleProtoTex checks raised {type(exc).__name__}: {exc}")

    try:
        proto_model = ProtoTEx(
            num_prototypes=4,
            num_pos_prototypes=2,
            n_classes=2,
            hidden_size=2,
            max_position_embeddings=2,
            vocab_size=16,
        )
        proto_model.bart_model.base_model.encoder = StaticEncoder(hidden)
        pos_protos = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ]
        )
        neg_protos = torch.tensor(
            [
                [[2.0, 0.0], [2.0, 1.0]],
                [[10.0, 0.0], [10.0, 0.0]],
            ]
        )
        with torch.no_grad():
            proto_model.pos_prototypes.copy_(pos_protos)
            proto_model.neg_prototypes.copy_(neg_protos)
            proto_model.classfn_model.weight.copy_(
                torch.tensor([[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]])
            )
            proto_model.classfn_model.bias.zero_()
        proto_result = proto_model(
            input_ids,
            attn_mask,
            labels,
            use_decoder=0,
            use_classfn=1,
            use_p1=1,
            use_p2=1,
            use_p3=1,
            distmask_lp1=1,
            distmask_lp2=1,
        )
        all_protos = torch.cat([pos_protos, neg_protos], dim=0)
        proto_distances = prototype_distance_matrix(hidden, all_protos)
        proto_mask = class_aware_distance_mask(labels, 4, 2)
        if proto_distances is None or proto_mask is None:
            skip_checks(4, "ProtoTEx helper distances or mask returned None")
        elif not isinstance(proto_result, tuple) or len(proto_result) != 2:
            skip_checks(4, "ProtoTEx did not return logits and losses")
        else:
            proto_logits, proto_losses = proto_result
            proto_masked = proto_distances + proto_mask
            expected_proto_logits = proto_model.classfn_model(proto_distances)
            expected_proto_lp1 = torch.mean(torch.min(proto_masked, dim=0)[0])
            expected_proto_lp2 = torch.mean(torch.min(proto_masked, dim=1)[0])
            expected_proto_lp3 = (1 / torch.sqrt(torch.tensor(2.0))) * torch.mean(torch.pdist(pos_protos.view(2, -1)))
            expected_proto_ce = F.cross_entropy(expected_proto_logits, labels)
            expected_proto_total = expected_proto_ce + 0.93 * expected_proto_lp1 + 0.92 * expected_proto_lp2 - expected_proto_lp3
            proto_losses_ok = isinstance(proto_losses, tuple) and len(proto_losses) > 5
            check("ProtoTEx logits use unmasked distances to positive and negative prototypes", close(proto_logits, expected_proto_logits), proto_logits)
            check("ProtoTEx masked lp1/lp2 use label-specific prototype partitions", proto_losses_ok and close(proto_losses[3], expected_proto_lp1) and close(proto_losses[4], expected_proto_lp2))
            check("ProtoTEx lp3 scales inter-positive-prototype distance", proto_losses_ok and close(proto_losses[5], expected_proto_lp3), proto_losses)
            check("ProtoTEx total loss combines class, reconstruction, lp1, lp2, and lp3 terms", proto_losses_ok and close(proto_losses[0], expected_proto_total), proto_losses)
    except Exception as exc:
        skip_checks(4, f"ProtoTEx checks raised {type(exc).__name__}: {exc}")

    try:
        status_model = ProtoTEx(4, 2, hidden_size=2, max_position_embeddings=2, vocab_size=16)
        status_model.set_encoder_status(True)
        encoder_flags = [all(parameter.requires_grad for parameter in layer.parameters()) for layer in status_model.bart_model.base_model.encoder.layers]
        status_model.set_decoder_status(False)
        decoder_flags = [any(parameter.requires_grad for parameter in layer.parameters()) for layer in status_model.bart_model.base_model.decoder.layers]
        status_model.set_protos_status("pos", False)
        status_model.set_classfn_status(True)
        check("encoder status leaves only the final encoder layer trainable", encoder_flags == [False, True], encoder_flags)
        check("decoder status can freeze every decoder layer", decoder_flags == [False, False], decoder_flags)
        check("prototype and classifier status toggles affect intended parameters", (not status_model.pos_prototypes.requires_grad) and status_model.neg_prototypes.requires_grad and all(parameter.requires_grad for parameter in status_model.classfn_model.parameters()))
    except Exception as exc:
        skip_checks(3, f"status toggle checks raised {type(exc).__name__}: {exc}")

    print(f"Passed {passed} checks; failed {failed} checks.")
    if failed != 0:
        raise SystemExit(1)
