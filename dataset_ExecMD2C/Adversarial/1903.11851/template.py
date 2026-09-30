import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ============================================================
# ground_truth.py - Feature Intertwiner Core Model Components
# Source: Adversarial/feature_intertwiner-master
#
# Contains only the core feature-intertwiner model components and direct
# dependencies: SAME padding, optimal transport module, FPN OT path, and
# classifier feature merge. No data loaders, training loops, inference
# pipelines, ROI extension builds, checkpoints, or remote assets.
# ============================================================

EPS = 1e-20


class AttrDict(dict):
    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )


def make_benchmark_config():
    config = AttrDict()
    config.DEV = AttrDict()
    config.DEV.OT_ONE_DIM_FORM = 'conv'
    config.DEV.LOSS_CHOICE = 'ot'
    config.DEV.CLS_MERGE_FEAT = True
    config.DEV.SWITCH = True
    config.DEV.STRUCTURE = 'beta'
    config.DEV.CLS_MERGE_MANNER = 'linear_add'
    config.DEV.CLS_MERGE_FAC = 0.4
    config.TRAIN = AttrDict()
    config.TRAIN.FPN_OT_LOSS = True
    config.DATA = AttrDict()
    config.DATA.IMAGE_SHAPE = [64, 64, 3]
    config.CTRL = AttrDict()
    config.CTRL.PHASE = 'train'
    return config


# --- [Original file: lib/sub_module.py] ---
class SamePad2d(nn.Module):
    """Mimic tensorflow's 'SAME' padding."""
    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


# --- [Original file: lib/OT_module.py] ---
class OptTrans(nn.Module):
    def __init__(self, config, ch_x, spatial_x=-1, ch_y=-1, spatial_y=-1,
                 epsilon=1., L=5, remove_bias=False, C_form='cosine', no_bp_P_L=True, skip_critic=False):

        super(OptTrans, self).__init__()
        self.config = config
        self.epsilon = 1./epsilon
        self.L = L
        self.remove_bias = remove_bias
        self.no_bp_P_L = no_bp_P_L
        self.C_form = C_form
        self.skip_critic = skip_critic
        self.two_dim = spatial_x > 1

        ch_y = ch_x if ch_y == -1 else ch_y
        spatial_y = spatial_x if spatial_y == -1 else spatial_y

        # define G_net
        if self.two_dim:
            if spatial_x != spatial_y:
                stride, out_pad = 2, 1   # upsample
            else:
                stride, out_pad = 1, 0   # keep spatial size unchanged
            _G_net_list = nn.ModuleList([
                nn.ConvTranspose2d(ch_x, ch_y, kernel_size=3, padding=1, stride=stride, output_padding=out_pad),
                nn.BatchNorm2d(ch_y),
                nn.ReLU(),
            ])
            self.G_net = nn.Sequential(*_G_net_list)
        else:
            self.G_net = nn.Sequential(*[
                nn.Conv1d(ch_x, ch_y, kernel_size=3, padding=1, stride=1),
                # nn.BatchNorm1d(ch_y),
                nn.ReLU()
            ])

        # define critic
        if not self.skip_critic:
            # this is a 2D case
            if self.two_dim:
                self.critic = nn.Sequential(*[
                    nn.Conv2d(ch_y, int(ch_y/2), kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(int(ch_y/2)),
                    nn.ReLU(),
                    nn.Conv2d(int(ch_y/2), int(ch_y/4), kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(int(ch_y/4),),
                    nn.ReLU(),
                ])
            else:
                # 1D case
                if self.config.DEV.OT_ONE_DIM_FORM == 'conv':
                    self.critic = nn.Sequential(*[
                        nn.Conv1d(ch_y, int(ch_y/4), kernel_size=3, padding=1, stride=1),
                        # comment BN layer; since 1(only one sample)x1024x1 will report error
                        # nn.BatchNorm1d(int(ch_y/4)),
                        nn.ReLU()
                    ])
                elif self.config.DEV.OT_ONE_DIM_FORM == 'fc':
                    self.critic = nn.Linear(ch_y, int(ch_y/8))

    def forward(self, x, y):
        """
        [TODO] Compute the feature-intertwiner optimal-transport loss between small and reliable features.

        Input:
            x: (batch, ch_x, spatial) or (batch, ch_x, h, w) - small or latent feature representation.
            y: (batch, ch_y, spatial) or (batch, ch_y, h, w) - reliable target feature representation.

        Output: (batch,) - per-sample OT-style discrepancy.

"""
        pass

    def _basic_compute_loss(self, x, y):
        """
        [TODO] Apply the critic and compute a batch of Sinkhorn transport costs.

        Input:
            x: (batch, channels, spatial...) - transformed source features.
            y: (batch, channels, spatial...) - target or comparison features.

        Output: (batch,) - one transport cost per sample.

"""
        pass

    def _sinkhorn_iterate(self, x, y):
        """
        [TODO] Compute one entropic optimal-transport cost with Sinkhorn iterations.

        Input:
            x: (num_points, feature_dim) - first point set.
            y: (num_points, feature_dim) - second point set.

        Output: scalar tensor - transport cost between the two point sets.

"""
        pass


# --- [Original file: lib/sub_module.py] ---
class FPN(nn.Module):
    def __init__(self, config, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.config = config
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 = nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

        if self.config.TRAIN.FPN_OT_LOSS:
            self.ot = True
            base_size = int(self.config.DATA.IMAGE_SHAPE[0] / 4)
            self.p2_ot = OptTrans(config, ch_x=256, spatial_x=base_size/2, spatial_y=base_size)
            self.p3_ot = OptTrans(config, ch_x=256, spatial_x=base_size/4, spatial_y=base_size/2)
            self.p4_ot = OptTrans(config, ch_x=256, spatial_x=base_size/8, spatial_y=base_size/4)
            # self.p5_ot = OptTrans(config, ch_x=256, spatial_x=base_size/16, spatial_y=base_size/8)
        else:
            self.ot = False

    def forward(self, x, mode):
        """
        [TODO] Build the FPN pyramid and attach cross-scale OT losses during training.

        Input:
            x: (batch, 3, height, width) - image tensor.
            mode: string - train enables OT supervision when configured.

        Output: list [P2, P3, P4, P5, P6, ot_loss], where P levels are
            (batch, out_channels, h_level, w_level) and ot_loss is (batch, 3).

"""
        pass


class Classifier(nn.Module):
    def __init__(self, depth, num_classes, pool_size, config):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.config = config
        self.merge_meta = config.DEV.CLS_MERGE_FEAT

        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)

        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)

        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, x, small_feat_input, small_gt_index, mode='train'):
        """
        [TODO] Run the Mask R-CNN classifier head with optional small-feature intertwining.

        Input:
            x: (num_rois, depth, pool_size, pool_size) - pooled ROI features.
            small_feat_input: (num_rois, 1024) - meta features extracted for small boxes.
            small_gt_index: (num_rois,) - class id or zero marker for each ROI.
            mode: string - preserved interface flag.

        Output:
            train/inference: [class_logits, class_probs, bbox_deltas]
                class_logits: (num_rois, num_classes)
                class_probs: (num_rois, num_classes)
                bbox_deltas: (num_rois, num_classes, 4)
            visualize phase: [intermediate_features, class_probs, bbox_deltas]

"""
        pass


# ============================================================
# __main__: Automated test suite for 5 ablated functions
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

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
    print("Feature Intertwiner: OT and meta-feature benchmark")
    print("Automated Test Suite - 5 ablated targets")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/5: OptTrans._sinkhorn_iterate
    # ==============================================================
    print("-" * 60)
    print("[Test 1/5] OptTrans._sinkhorn_iterate - transport plan cost")
    if not torch.cuda.is_available():
        skip_checks(5, "source implementation constructs Sinkhorn variables on CUDA")
    else:
        try:
            config = make_benchmark_config()
            module = OptTrans(config, ch_x=8, spatial_x=-1, epsilon=1.0, L=3).to(device)
            x_base = torch.randn(4, 6, device=device, requires_grad=True)
            x = x_base * 1.0
            y = x.detach().clone()
            z = torch.randn(4, 6, device=device)
            same_loss = module._sinkhorn_iterate(x, y)
            diff_loss = module._sinkhorn_iterate(x, z)
            check("sinkhorn output not None", same_loss is not None)
            if same_loss is not None:
                check("sinkhorn scalar output", tuple(same_loss.shape) == (),
                      f"expected scalar, got {tuple(same_loss.shape)}")
                check("sinkhorn finite", torch.isfinite(same_loss).item() and torch.isfinite(diff_loss).item())
                check("sinkhorn identical lower cost", same_loss.item() <= diff_loss.item() + 1e-4)
                check("sinkhorn preserves differentiable output", same_loss.requires_grad)
            else:
                skip_checks(4, "OptTrans._sinkhorn_iterate returned None")
        except Exception as e:
            skip_checks(5, f"OptTrans._sinkhorn_iterate raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/5: OptTrans._basic_compute_loss
    # ==============================================================
    print("-" * 60)
    print("[Test 2/5] OptTrans._basic_compute_loss - critic-space OT batch loss")
    if not torch.cuda.is_available():
        skip_checks(4, "source implementation constructs Sinkhorn variables on CUDA")
    else:
        try:
            config = make_benchmark_config()
            module = OptTrans(config, ch_x=8, spatial_x=-1, epsilon=1.0, L=2).to(device)
            x = torch.randn(2, 8, 5, device=device, requires_grad=True)
            y = torch.randn(2, 8, 5, device=device)
            loss = module._basic_compute_loss(x, y)
            check("basic OT output not None", loss is not None)
            if loss is not None:
                check("basic OT batch shape", tuple(loss.shape) == (2,),
                      f"expected (2,), got {tuple(loss.shape)}")
                check("basic OT finite", torch.isfinite(loss).all().item())
                check("basic OT preserves differentiable output", loss.requires_grad)
            else:
                skip_checks(3, "OptTrans._basic_compute_loss returned None")
        except Exception as e:
            skip_checks(4, f"OptTrans._basic_compute_loss raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/5: OptTrans.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/5] OptTrans.forward - generator plus bias-corrected OT loss")
    if not torch.cuda.is_available():
        skip_checks(5, "source implementation constructs Sinkhorn variables on CUDA")
    else:
        try:
            config = make_benchmark_config()
            module = OptTrans(config, ch_x=8, spatial_x=-1, epsilon=1.0, L=2, remove_bias=False).to(device)
            x = torch.randn(2, 8, 5, device=device, requires_grad=True)
            y = torch.randn(2, 8, 5, device=device)
            loss = module(x, y)
            check("OptTrans.forward output not None", loss is not None)
            if loss is not None:
                check("OptTrans.forward batch shape", tuple(loss.shape) == (2,),
                      f"expected (2,), got {tuple(loss.shape)}")
                check("OptTrans.forward finite", torch.isfinite(loss).all().item())
                check("OptTrans.forward generator shape", tuple(module.G_net(x.detach()).shape) == (2, 8, 5))
                check("OptTrans.forward preserves differentiable output", loss.requires_grad)
            else:
                skip_checks(4, "OptTrans.forward returned None")
        except Exception as e:
            skip_checks(5, f"OptTrans.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/5: FPN.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/5] FPN.forward - top-down pyramid with OT supervision")
    if not torch.cuda.is_available():
        skip_checks(5, "source implementation initializes FPN OT loss on CUDA")
    else:
        try:
            config = make_benchmark_config()
            C1 = nn.Sequential(nn.Conv2d(3, 256, 1), nn.ReLU()).to(device)
            C2 = nn.Sequential(nn.Conv2d(256, 256, 1), nn.ReLU()).to(device)
            C3 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU()).to(device)
            C4 = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=2, padding=1), nn.ReLU()).to(device)
            C5 = nn.Sequential(nn.Conv2d(1024, 2048, 3, stride=2, padding=1), nn.ReLU()).to(device)
            module = FPN(config, C1, C2, C3, C4, C5, out_channels=256).to(device)
            x = torch.randn(2, 3, 16, 16, device=device)
            outputs = module(x, mode="train")
            check("FPN output not None", outputs is not None)
            if outputs is not None:
                p2, p3, p4, p5, p6, ot_loss = outputs
                check("FPN returns six outputs", len(outputs) == 6)
                check("FPN p2 shape", tuple(p2.shape) == (2, 256, 16, 16),
                      f"expected (2, 256, 16, 16), got {tuple(p2.shape)}")
                finite = all(torch.isfinite(t).all().item() for t in [p2, p3, p4, p5, p6, ot_loss])
                check("FPN outputs finite", finite)
                check("FPN OT loss shape", tuple(ot_loss.shape) == (2, 3),
                      f"expected (2, 3), got {tuple(ot_loss.shape)}")
            else:
                skip_checks(4, "FPN.forward returned None")
        except Exception as e:
            skip_checks(5, f"FPN.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 5/5: Classifier.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 5/5] Classifier.forward - small-feature merge into classifier")
    try:
        config = make_benchmark_config()
        module = Classifier(depth=256, num_classes=5, pool_size=7, config=config).to(device)
        module.eval()
        x = torch.randn(3, 256, 7, 7, device=device)
        small_feat = torch.randn(3, 1024, device=device)
        small_gt = torch.tensor([1, 0, 3], device=device)
        with torch.no_grad():
            outputs = module(x, small_feat, small_gt, mode="train")
        check("Classifier output not None", outputs is not None)
        if outputs is not None:
            logits, probs, bbox = outputs
            check("Classifier logits shape", tuple(logits.shape) == (3, 5),
                  f"expected (3, 5), got {tuple(logits.shape)}")
            check("Classifier probs finite", torch.isfinite(probs).all().item())
            check("Classifier probabilities sum to one", torch.allclose(probs.sum(dim=1), torch.ones(3, device=device), atol=1e-5))
            check("Classifier bbox shape", tuple(bbox.shape) == (3, 5, 4),
                  f"expected (3, 5, 4), got {tuple(bbox.shape)}")
            config.DEV.CLS_MERGE_FEAT = False
            no_merge = Classifier(depth=256, num_classes=5, pool_size=7, config=config).to(device)
            no_merge.load_state_dict(module.state_dict())
            no_merge.eval()
            with torch.no_grad():
                logits_no_merge = no_merge(x, small_feat, small_gt, mode="train")[0]
            check("Classifier merge changes positive-gt logits", not torch.allclose(logits[[0, 2]], logits_no_merge[[0, 2]]))
        else:
            skip_checks(5, "Classifier.forward returned None")
    except Exception as e:
        skip_checks(6, f"Classifier.forward raised {type(e).__name__}: {e}")
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
