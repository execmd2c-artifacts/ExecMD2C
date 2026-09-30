# ============================================================
# ground_truth.py - RIDE Core Model Components (Self-contained)
# Source:
#   Miscellaneous/RIDE-LongTailRecognition-main/model/ldam_drw_resnets/ride_resnet_cifar.py
#   Miscellaneous/RIDE-LongTailRecognition-main/model/loss.py
#
# Contains ONLY model architecture components and the core RIDE loss.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


# --- [Original file: model/ldam_drw_resnets/ride_resnet_cifar.py] ---
__all__ = ['ResNet_s']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.planes = planes
                self.in_planes = in_planes
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_experts, num_classes=10, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, use_experts=None, s=30):
        super(ResNet_s, self).__init__()
        
        self.in_planes = 16
        self.num_experts = num_experts

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.in_planes = self.next_in_planes

        if layer2_output_dim is None:
            if reduce_dimension:
                layer2_output_dim = 24
            else:
                layer2_output_dim = 32

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 48
            else:
                layer3_output_dim = 64

        self.layer2s = nn.ModuleList([self._make_layer(block, layer2_output_dim, num_blocks[1], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        
        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(layer3_output_dim, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList([nn.Linear(layer3_output_dim, num_classes) for _ in range(num_experts)])
            s = 1

        if use_experts is None:
            self.use_experts = list(range(num_experts))
        elif use_experts == "rand":
            self.use_experts = None
        else:
            self.use_experts = [int(item) for item in use_experts.split(",")]

        self.s = s

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _separate_part(self, x, ind):
        out = x
        out = (self.layer2s[ind])(out)
        out = (self.layer3s[ind])(out)
        self.feat_before_GAP.append(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        self.feat.append(out)
        out = (self.linears[ind])(out)
        out = out * self.s
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        
        outs = []
        self.feat = []
        self.logits = outs
        self.feat_before_GAP = []
        
        if self.use_experts is None:
            use_experts = random.sample(range(self.num_experts), self.num_experts - 1)
        else:
            use_experts = self.use_experts
        
        for ind in use_experts:
            outs.append(self._separate_part(out, ind))
        self.feat = torch.stack(self.feat, dim=1)
        self.feat_before_GAP = torch.stack(self.feat_before_GAP, dim=1)
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        return final_out


# --- [Original file: model/loss.py] ---
eps = 1e-7

class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss


# ============================================================
# __main__: Automated test suite for 3 ablated functions
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

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
    print("RIDE: Long-tailed Recognition by Routing Diverse Distribution-Aware Experts")
    print("Automated reproduction benchmark - 3 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/3: ResNet_s._separate_part
    # ==============================================================
    print("-" * 60)
    print("[Test 1/3] ResNet_s._separate_part - single expert branch")
    try:
        model = ResNet_s(BasicBlock, [1, 1, 1], num_experts=3, num_classes=5, layer2_output_dim=16, layer3_output_dim=16).to(device)
        model.eval()
        x = torch.randn(2, 3, 32, 32, device=device)
        with torch.no_grad():
            shared = F.relu(model.bn1(model.conv1(x)))
            shared = model.layer1(shared)
            model.feat = []
            model.feat_before_GAP = []
            expert_logits = model._separate_part(shared, 1)
        check("expert branch output not None", expert_logits is not None)
        if expert_logits is not None:
            feat_shape_ok = len(model.feat) == 1 and tuple(model.feat[0].shape) == (2, 16)
            feat_map_shape_ok = len(model.feat_before_GAP) == 1 and tuple(model.feat_before_GAP[0].shape) == (2, 16, 8, 8)
            check("expert branch logits shape", tuple(expert_logits.shape) == (2, 5), f"expected (2, 5), got {tuple(expert_logits.shape)}")
            check("expert branch outputs finite", torch.isfinite(expert_logits).all().item())
            check("expert branch records pooled and pre-GAP features", feat_shape_ok and feat_map_shape_ok)
        else:
            skip_checks(3, "ResNet_s._separate_part returned None")
    except Exception as e:
        skip_checks(4, f"ResNet_s._separate_part raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/3: ResNet_s.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/3] ResNet_s.forward - multi-expert routing and mean fusion")
    try:
        model = ResNet_s(BasicBlock, [1, 1, 1], num_experts=3, num_classes=5, layer2_output_dim=16, layer3_output_dim=16).to(device)
        model.eval()
        x = torch.randn(2, 3, 32, 32, device=device)
        with torch.no_grad():
            output = model(x)
        check("RIDE forward output not None", output is not None)
        if output is not None:
            logits_stack = torch.stack(model.logits, dim=1)
            mean_matches = torch.allclose(output, logits_stack.mean(dim=1), atol=1e-6)
            feat_ok = tuple(model.feat.shape) == (2, 3, 16)
            pre_gap_ok = tuple(model.feat_before_GAP.shape) == (2, 3, 16, 8, 8)
            check("RIDE forward output shape", tuple(output.shape) == (2, 5), f"expected (2, 5), got {tuple(output.shape)}")
            check("RIDE forward output finite", torch.isfinite(output).all().item())
            check("RIDE forward averages all expert logits and stacks features", mean_matches and feat_ok and pre_gap_ok)
        else:
            skip_checks(3, "ResNet_s.forward returned None")
    except Exception as e:
        skip_checks(4, f"ResNet_s.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/3: RIDELoss.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/3] RIDELoss.forward - per-expert loss with diversity regularization")
    try:
        output_logits = torch.tensor([[2.0, 0.2, -0.3], [0.1, 1.5, -0.4]], dtype=torch.float32, device=device, requires_grad=True)
        expert_logits = [
            torch.tensor([[2.3, 0.0, -0.4], [0.0, 1.2, -0.2]], dtype=torch.float32, device=device, requires_grad=True),
            torch.tensor([[1.5, 0.6, -0.1], [0.4, 1.9, -0.8]], dtype=torch.float32, device=device, requires_grad=True),
        ]
        target = torch.tensor([0, 1], dtype=torch.long, device=device)
        loss_fn = RIDELoss(cls_num_list=None, base_diversity_temperature=2.0, base_loss_factor=1.25, additional_diversity_factor=-0.2)
        loss_fn.per_cls_weights_diversity = None
        extra_info = {"logits": expert_logits}
        loss = loss_fn(output_logits, target, extra_info=extra_info)
        check("RIDELoss output not None", loss is not None)
        if loss is not None:
            ce_sum = sum(F.cross_entropy(item, target) for item in expert_logits) * 1.25
            check("RIDELoss output is scalar", loss.dim() == 0, f"expected scalar, got shape {tuple(loss.shape)}")
            check("RIDELoss output finite", torch.isfinite(loss).item())
            check("RIDELoss includes nonzero diversity adjustment", not torch.allclose(loss, ce_sum, atol=1e-6))
            loss.backward()
            grads_ok = all(item.grad is not None and item.grad.abs().sum().item() > 0 for item in expert_logits)
            check("RIDELoss sends gradients to every expert logits tensor", grads_ok)
        else:
            skip_checks(4, "RIDELoss.forward returned None")
    except Exception as e:
        skip_checks(5, f"RIDELoss.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Final Score
    # ==============================================================
    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some ablated functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
