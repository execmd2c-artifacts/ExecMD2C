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
    """
    TODO: Reproduce ProtoTEx's flattened prototype-distance computation.

    The implementation should validate that encoded examples and prototypes are both
    rank-3 tensors with matching [sequence, hidden] dimensions, flatten each example
    and prototype tensor into one vector, and return a [batch, num_prototypes] matrix
    of Euclidean distances using torch.cdist.
    """
    pass


def class_aware_distance_mask(
    labels: torch.Tensor,
    num_prototypes: int,
    num_pos_prototypes: int,
    device: Optional[torch.device] = None,
    high_value: float = 1e7,
) -> torch.Tensor:
    """
    TODO: Reproduce the class-specific prototype masking used for Lp1/Lp2.

    Build the same two-row distance grounder as ProtoTEx: label 0 receives a very
    large value on positive prototypes, while label 1 receives a very large value on
    negative prototypes. Return the rows indexed by the provided labels on the
    requested device.
    """
    pass


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
        """
        TODO: Reproduce SimpleProtoTex's forward pass.

        Encode the input with either the seq2seq path or encoder-only path, compute
        flattened prototype distances when class/p1/p2 terms are requested, derive
        Lp1 over prototypes and Lp2 over examples, classify from the distance matrix,
        optionally suppress reconstruction loss when use_rc is false, and return the
        logits plus the five-loss tuple in the same order as the answer key.
        """
        pass


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
        """
        TODO: Reproduce ProtoTEx's prototype-tensor forward pass.

        Concatenate positive and negative prototypes, compute flattened distances,
        optionally instance-normalize the distance matrix, build label-aware masked
        distances for Lp1/Lp2, compute positive-prototype separation Lp3, classify
        with the optional dropout/special-classifier branches, suppress reconstruction
        loss when requested, and return logits plus the six-loss tuple in the answer
        key order.
        """
        pass


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
