import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class CutsModel(nn.Module):

    def __init__(self, num_classes, input_size, num_layers):
        '''
        Args:
            num_classes (int): The number of labels to classify
            input_size (int): The size of each feature
            
        '''
        super().__init__()
        logging.info(f'Model: num_classes {num_classes} input_size {input_size} num_layers {num_layers} ')

        self.num_classes = num_classes
        self.input_size = input_size
        self.num_layers = num_layers
        self.feat_transform, mid_feat_size = self._create_sequential_linear_relu_layers(self.num_layers, self.input_size)
        self.relu = nn.ReLU()
        self.fc_class = nn.Linear(mid_feat_size, self.num_classes)

    @staticmethod
    @staticmethod
    def _create_sequential_linear_relu_layers(num_layers, input_size, output_size=None, reduction_factor=2):
        """
        TODO: Reproduce the sequential linear/ReLU feature-transform builder.

        Inputs:
            num_layers: number of linear stages to create.
            input_size: feature dimension entering the first layer.
            output_size: optional required final feature dimension.
            reduction_factor: factor used to shrink hidden dimensions.

        Output:
            A tuple (sequential_module, final_feature_size).

        Required behavior:
            Build an ordered stack of linear layers and ReLU activations that progressively reduces
            feature dimensionality. When output_size is provided, keep intermediate dimensions no
            smaller than the requested output and make the final linear layer produce exactly that
            output size. When output_size is absent, the final activation should be omitted and the
            reported final feature size must match the last linear layer output. Preserve deterministic
            layer naming and ordering so downstream checkpoints and tests can address modules by name.
        """
        pass

    def forward(self, features):
        """
        TODO: Reproduce the LearningToCut model forward pass.

        Input:
            features: tensor of snippet features with shape (N, input_size).

        Output:
            Tuple (logits, transformed_features), where logits has shape (N, num_classes) and
            transformed_features has shape (N, hidden_dim).

        Required behavior:
            Apply the feature transformation stack, keep the transformed representation for the loss,
            apply the model ReLU before classification, and return both classification logits and the
            pre-classifier transformed features. Preserve gradient flow from both returned tensors back
            to the input features and model parameters.
        """
        pass


class CutsLoss():
    def __init__(self, contrastive_loss_type, alpha_contrastive, alpha_ce_pairs, bce_pairs_logits=False, alpha_ce=0, boundary_oracle=False):
        '''
        Args:
            Contrastive loss type (str): type of contrastive loss
            alpha_contrastive (float): Weight for the loss
            bce_pairs_logits compute the bce from logits not from features
        '''

        self.alpha_contrastive = alpha_contrastive
        self.alpha_ce = alpha_ce
        self.alpha_ce_pairs = alpha_ce_pairs
        self.bce_pairs_logits = bce_pairs_logits
        self.contrastive_loss_type = contrastive_loss_type
        self.boundary_oracle = boundary_oracle
        
        if self.bce_pairs_logits:
            self.beta_pairs = 0.5
        else:
            self.beta_pairs = 0.005

        if self.contrastive_loss_type == 'triplet':
            self.contrastive_loss = self.triplet_loss
            self.beta_contrastive = 1
        elif self.contrastive_loss_type == 'nce':
            self.contrastive_loss = self.nce_loss
            self.beta_contrastive = 0.2
        else:
            raise IOError(f'Invalid Loss type. Error message: {e}')

    def compute_loss(self, logits, features, targets, device):

        loss_results = {'cross_entropy_loss': torch.tensor(0.0).to(device),
                        'constrastive_loss': torch.tensor(0.0).to(device),
                        'pairs_class_loss': torch.tensor(0.0).to(device),
                        'loss': torch.tensor(0.0).to(device),
                        }
        num_valid_videos = 0
        for i, video_name in enumerate(targets['video-names']):
            this_pairs = targets['video-name-to-pairs'][video_name]
            if len(this_pairs) == 0:
                continue
            num_valid_videos =+ 1
            start, end = targets['video-name-to-slice'][video_name]
            this_logits = logits[start:end].squeeze(-1)
            this_features = features[start:end]
            this_labels = targets['labels'][start:end]
            this_masks_shot_idx = targets['masks_shot_idx'][start:end]
            backprop_idx = this_labels >= 0
            this_loss_results = self._loss_for_one_video(logits=this_logits,
                                                        features=this_features,
                                                        labels=this_labels,
                                                        backprop_idx=backprop_idx,
                                                        masks_shot_idx=this_masks_shot_idx,
                                                        pairs=this_pairs,
                                                        snippet_weight=targets['snippet_class_weight'],
                                                        pairs_weight=targets['pairs_class_weight'],
                                                        device=device)
            for k, v in this_loss_results.items():
                loss_results[k] += v

        for k, v in loss_results.items():
            loss_results[k] /= len(targets['video-names'])

        return loss_results

    def _loss_for_one_video(self, logits, features, labels, backprop_idx, masks_shot_idx, pairs, snippet_weight, pairs_weight, device):
        '''
            TODO: Specify here the dimensions
            cas: 1 x T x num_classes
            attention" 1 x T x {1,2}
            labels: 1 x 1
            pseudo_gt: 1 x T
            bg_cas: 1 x T X num_classes+1
        '''
        snippet_class_loss = F.binary_cross_entropy_with_logits(input=logits[backprop_idx].view(1, -1),
                                                    target=labels[backprop_idx].unsqueeze(0).type(torch.cuda.FloatTensor).to(device),
                                                    pos_weight=torch.tensor(snippet_weight).to(device))
        
        if self.bce_pairs_logits:
            scores = torch.matmul(logits.unsqueeze(1),logits.unsqueeze(0))/2
        else:
            scores = torch.matmul(features, features.transpose(0,1))
        
        pairs_class_loss = F.binary_cross_entropy_with_logits(input=scores[pairs[:, 0], pairs[:, 1]].to(device),
                                                            target=pairs[:,2].type(torch.cuda.FloatTensor).to(device),
                                                            pos_weight=torch.tensor(pairs_weight).to(device))

        sim_loss = 0
        num_valid_shots = 0
        # Find triplet loss per shot
        for shot_idx in range(1,max(masks_shot_idx)):
            this_idx_shot = torch.where(masks_shot_idx == shot_idx)[0]
            next_idx_shot = torch.where(masks_shot_idx == shot_idx +1)[0]
            # Check if the two shots are adjacent or not, they have to have two zeros between them
            if next_idx_shot[0] - this_idx_shot[-1] != 3:
                continue
            this_features = features[masks_shot_idx == shot_idx]
            neighbour_features = features[masks_shot_idx == shot_idx+1]
            if neighbour_features.shape[0]<=1:
                continue
            num_valid_shots += 1
            positive_pair = (this_features[-1], neighbour_features[0])
            
            if self.boundary_oracle:
                anchor_features = this_features[-1:]
            else:
                anchor_features = this_features

            sim_loss += self.contrastive_loss(positive_pair, anchor_features, neighbour_features[1:])

        contrastive_loss = sim_loss/max(1,num_valid_shots) 

        final_contrastive_loss = self.alpha_contrastive * self.beta_contrastive * contrastive_loss
        final_shot_class_loss = self.alpha_ce * snippet_class_loss
        final_pair_class_loss = self.alpha_ce_pairs* self.beta_pairs * pairs_class_loss

        loss = final_contrastive_loss + final_shot_class_loss + final_pair_class_loss

        loss_results = {'cross_entropy_loss': snippet_class_loss,
                        'constrastive_loss': contrastive_loss,
                        'pairs_class_loss': pairs_class_loss,
                        'loss': loss}

        return loss_results

    def triplet_loss(self, positive_pair, anchor_features, negative_features, triplet_margin=1):
        pos_distance = torch.norm(positive_pair[0] - positive_pair[1], dim=-1)
        neg_distances = torch.norm(anchor_features.unsqueeze(0) - negative_features.unsqueeze(1), dim=-1).view(-1)    
        triplet_loss = nn.functional.relu((pos_distance - neg_distances) + triplet_margin).mean()
        return triplet_loss

    def nce_loss(self, positive_pair, anchor_features, negative_features):
        """
        TODO: Reproduce the normalized contrastive loss used by LearningToCut.

        Inputs:
            positive_pair: two feature vectors representing adjacent positive snippets, each with shape (C).
            anchor_features: anchor snippet features with shape (A, C).
            negative_features: negative snippet features with shape (K, C).

        Output:
            A scalar tensor loss.

        Required behavior:
            Normalize positive, anchor, and negative feature vectors by their L2 norms, compute an
            exponentiated positive similarity score and exponentiated anchor-negative similarities,
            aggregate the negative evidence, and return the negative log probability of the positive
            match against positive-plus-negative evidence. The result must remain differentiable with
            respect to all feature tensors that require gradients.
        """
        pass


if __name__ == "__main__":
    import traceback

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

    print("Running LearningToCut benchmark checks...")
    torch.manual_seed(11)

    try:
        model = CutsModel(num_classes=1, input_size=8, num_layers=3)
        features = torch.randn(6, 8, requires_grad=True)
        logits, transformed = model(features)
        check("model_forward_functionality", logits is not None and transformed is not None)
        if logits is None or transformed is None:
            skip_checks(3, "CutsModel.forward returned None")
        else:
            check("model_forward_shapes", tuple(logits.shape) == (6, 1) and transformed.shape[0] == 6,
                  f"logits={tuple(logits.shape)}, transformed={tuple(transformed.shape)}")
            check("model_forward_finite", torch.isfinite(logits).all().item() and torch.isfinite(transformed).all().item())
            (logits.sum() + transformed.sum()).backward()
            grad_ok = features.grad is not None and torch.isfinite(features.grad).all().item() and features.grad.abs().sum().item() > 0
            check("model_forward_gradient", grad_ok)
    except Exception as exc:
        traceback.print_exc()
        skip_checks(4, f"CutsModel checks raised {type(exc).__name__}: {exc}")

    try:
        seq, out_size = CutsModel._create_sequential_linear_relu_layers(
            num_layers=3,
            input_size=16,
            output_size=4,
            reduction_factor=2,
        )
        layer_names = list(seq._modules.keys())
        helper_input = torch.randn(5, 16)
        helper_output = seq(helper_input)
        check("layer_builder_functionality", helper_output is not None)
        if helper_output is None:
            skip_checks(2, "layer builder returned None output")
        else:
            check("layer_builder_shape", tuple(helper_output.shape) == (5, 4) and out_size == 4,
                  f"output={tuple(helper_output.shape)}, out_size={out_size}")
            expected_names = ['linear_0', 'relu_0', 'linear_1', 'relu_1', 'linear_2']
            check("layer_builder_order", layer_names == expected_names, str(layer_names))
    except Exception as exc:
        traceback.print_exc()
        skip_checks(3, f"layer builder checks raised {type(exc).__name__}: {exc}")

    try:
        loss_obj = CutsLoss(
            contrastive_loss_type='nce',
            alpha_contrastive=1.0,
            alpha_ce_pairs=0.0,
            bce_pairs_logits=False,
            alpha_ce=0.0,
            boundary_oracle=False,
        )
        positive_pair = (torch.tensor([1.0, 0.0, 0.0], requires_grad=True),
                         torch.tensor([0.9, 0.1, 0.0], requires_grad=True))
        anchor_features = torch.stack([positive_pair[0], torch.tensor([0.8, 0.2, 0.0])])
        negative_features = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True)
        nce = loss_obj.nce_loss(positive_pair, anchor_features, negative_features)
        check("nce_loss_functionality", nce is not None)
        if nce is None:
            skip_checks(2, "nce_loss returned None")
        else:
            check("nce_loss_scalar_finite", nce.ndim == 0 and torch.isfinite(nce).item(), f"shape={tuple(nce.shape)}")
            nce.backward()
            grad_ok = positive_pair[0].grad is not None and negative_features.grad is not None
            check("nce_loss_gradient", grad_ok)
    except Exception as exc:
        traceback.print_exc()
        skip_checks(3, f"nce_loss checks raised {type(exc).__name__}: {exc}")

    print(f"Checks passed: {passed}")
    print(f"Checks failed: {failed}")
    if failed != 0:
        raise SystemExit(1)
