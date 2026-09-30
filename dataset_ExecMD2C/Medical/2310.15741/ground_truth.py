"""
ground_truth.py for Proto-Caps core model components.

Source-consolidated from:
- capsulelayers.py
- models.py
- loss.py

Only the capsule layers, ProtoCapsNet architecture, prototype-distance
computation, and direct prototype/attribute loss functions are included. Data
preparation, parameter-update procedures, prototype maintenance procedures,
evaluation scripts, saved-state I/O, and CLI code are intentionally excluded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# --- [Original file: capsulelayers.py] ---
def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of capsules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3, activation_fn="softmax"):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
        self.activation_fn = activation_fn

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            if self.activation_fn == "softmax":
                c = F.softmax(b, dim=1)
            elif self.activation_fn == "sigmoid":
                c = torch.sigmoid(b)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    """
    Apply Conv with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """

    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, threeD, stride=1, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_caps = dim_caps
        if threeD:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = self.conv(x)
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        return squash(outputs)


# --- [Original file: models.py] ---
class ProtoCapsNet(nn.Module):
    def __init__(self, input_size, numcaps, routings, out_dim_caps, activation_fn, threeD, numProtos):
        super(ProtoCapsNet, self).__init__()
        self.input_size = input_size
        self.numcaps = numcaps
        self.routings = routings
        self.out_dim_caps = out_dim_caps
        self.numclasses = 5
        self.threeD = threeD
        self.numProtos = numProtos

        # Layer 1: Just a conventional Conv2D layer
        if self.threeD:
            self.conv1 = nn.Conv3d(input_size[0], 256, kernel_size=9, stride=1, padding=0)
        else:
            self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=9, threeD=self.threeD, stride=2, padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        if self.threeD:
            self.digitcaps = DenseCapsule(in_num_caps=131072, in_dim_caps=8,
                                          out_num_caps=numcaps, out_dim_caps=out_dim_caps, routings=routings,
                                          activation_fn=activation_fn)
        else:
            self.digitcaps = DenseCapsule(in_num_caps=32 * 8 * 8, in_dim_caps=8,
                                          out_num_caps=numcaps, out_dim_caps=out_dim_caps, routings=routings,
                                          activation_fn=activation_fn)
        # Decoder network.
        if self.threeD:
            self.decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(numcaps * out_dim_caps, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, input_size[0] * input_size[1] * input_size[2] * input_size[3]),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(numcaps * out_dim_caps, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
                nn.Sigmoid()
            )

        # Prediction layers
        self.predOutLayers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=numcaps * out_dim_caps, out_features=self.numclasses),
            nn.Softmax(dim=-1)
        )

        self.relu = nn.ReLU()

        # Attribute layers
        self.attrOutLayer0 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer1 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer2 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer3 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer4 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer5 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer6 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )
        self.attrOutLayer7 = nn.Sequential(
            nn.Linear(in_features=out_dim_caps, out_features=1),
            nn.Sigmoid()
        )

        # Prototype vectors
        self.protodigis0 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis1 = nn.Parameter(torch.rand((4, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis2 = nn.Parameter(torch.rand((6, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis3 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis4 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis5 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis6 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis7 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
        self.protodigis_list = [self.protodigis0, self.protodigis1, self.protodigis2, self.protodigis3,
                                self.protodigis4, self.protodigis5, self.protodigis6, self.protodigis7]

    def forwardCapsule(self, x_ex):
        capsout0 = self.attrOutLayer0(x_ex[:, 0, :])
        capsout1 = self.attrOutLayer1(x_ex[:, 1, :])
        capsout2 = self.attrOutLayer2(x_ex[:, 2, :])
        capsout3 = self.attrOutLayer3(x_ex[:, 3, :])
        capsout4 = self.attrOutLayer4(x_ex[:, 4, :])
        capsout5 = self.attrOutLayer5(x_ex[:, 5, :])
        capsout6 = self.attrOutLayer6(x_ex[:, 6, :])
        capsout7 = self.attrOutLayer7(x_ex[:, 7, :])
        pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5, capsout6, capsout7), dim=1)
        reconstruction = self.decoder(x_ex)
        pred = self.predOutLayers(x_ex)

        return pred, pred_attr, reconstruction.view(-1, *self.input_size)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)

        # attribute out
        capsout0 = self.attrOutLayer0(x[:, 0, :])
        capsout1 = self.attrOutLayer1(x[:, 1, :])
        capsout2 = self.attrOutLayer2(x[:, 2, :])
        capsout3 = self.attrOutLayer3(x[:, 3, :])
        capsout4 = self.attrOutLayer4(x[:, 4, :])
        capsout5 = self.attrOutLayer5(x[:, 5, :])
        capsout6 = self.attrOutLayer6(x[:, 6, :])
        capsout7 = self.attrOutLayer7(x[:, 7, :])
        pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5, capsout6, capsout7), dim=1)

        # reconstruction
        reconstruction = self.decoder(x)

        # prediction out
        pred = self.predOutLayers(x)

        dists_to_protos = self.getDistance(x)

        return pred, pred_attr, reconstruction.view(-1, *self.input_size), dists_to_protos

    def getDistance(self, x):
        """
        Capsule wise calculation of distance to closest prototype vector
        :param x: vectors to calculate distance to
        :return: distances to closest protoype vector
        """

        xreshaped = torch.unsqueeze(x, dim=1)
        xreshaped = torch.unsqueeze(xreshaped, dim=1)
        protoreshaped0 = torch.unsqueeze(self.protodigis0, dim=0)
        protoreshaped1 = torch.unsqueeze(self.protodigis1, dim=0)
        protoreshaped2 = torch.unsqueeze(self.protodigis2, dim=0)
        protoreshaped3 = torch.unsqueeze(self.protodigis3, dim=0)
        protoreshaped4 = torch.unsqueeze(self.protodigis4, dim=0)
        protoreshaped5 = torch.unsqueeze(self.protodigis5, dim=0)
        protoreshaped6 = torch.unsqueeze(self.protodigis6, dim=0)
        protoreshaped7 = torch.unsqueeze(self.protodigis7, dim=0)

        dists_0 = (xreshaped[:, :, :, 0, :] - protoreshaped0).pow(2).sum(-1).sqrt()

        dists_1 = (xreshaped[:, :, :, 1, :] - protoreshaped1).pow(2).sum(-1).sqrt()

        dists_2 = (xreshaped[:, :, :, 2, :] - protoreshaped2).pow(2).sum(-1).sqrt()

        dists_3 = (xreshaped[:, :, :, 3, :] - protoreshaped3).pow(2).sum(-1).sqrt()

        dists_4 = (xreshaped[:, :, :, 4, :] - protoreshaped4).pow(2).sum(-1).sqrt()

        dists_5 = (xreshaped[:, :, :, 5, :] - protoreshaped5).pow(2).sum(-1).sqrt()

        dists_6 = (xreshaped[:, :, :, 6, :] - protoreshaped6).pow(2).sum(-1).sqrt()

        dists_7 = (xreshaped[:, :, :, 7, :] - protoreshaped7).pow(2).sum(-1).sqrt()

        dists_to_protos = [dists_0, dists_1, dists_2, dists_3, dists_4, dists_5, dists_6, dists_7]

        return dists_to_protos


# --- [Original file: loss.py] ---
def xcaps_loss(y, y_pred, x, x_recon, lam_recon, attr_gt, attr_pred, lam_attr, epoch,
               dists_to_protos, sample_id, idx_with_attri, max_dist, arguments):
    L_recon = nn.MSELoss()(x_recon, x)

    L_pred = nn.KLDivLoss(reduction="batchmean")(y_pred, y)

    batchidx_with_attri = []
    for i in range(len(sample_id)):
        if sample_id[i] in idx_with_attri:
            batchidx_with_attri.append(i)
    L_attr = 0.0
    for i in range(attr_gt.shape[-1]):
        L_attr += nn.MSELoss()(attr_pred[batchidx_with_attri, i], attr_gt[batchidx_with_attri, i])

    L_sep = 0.0
    L_cluster_allmean = 0.0
    L_cluster_allcpsi = 0.0

    if epoch < arguments.warmup:
        total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr
    else:
        if len(batchidx_with_attri) > 0:
            for capsule_idx in range(len(dists_to_protos)):
                if capsule_idx in [0, 3, 4, 5, 6, 7]:
                    idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.125).nonzero()
                    idx1 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.125) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.375)).nonzero()
                    idx2 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.375) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.625)).nonzero()
                    idx3 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.625) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.875)).nonzero()
                    idx4 = (attr_gt[batchidx_with_attri, capsule_idx] >= 0.875).nonzero()
                    idxs = [idx0, idx1, idx2, idx3, idx4]
                    L_cluster = 0

                    for idxi in range(len(idxs)):
                        if len(idxs[idxi]) > 0:
                            L_cluster += torch.sum(
                                torch.min(torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]),dim=-1)[
                                    0])

                    L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                    L_sep_loss = sep_loss(max_dist=max_dist, selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],indices=[idx0,idx1,idx2,idx3,idx4])

                    L_sep += L_sep_loss / (len(batchidx_with_attri) * 4)

                elif capsule_idx == 1:
                    idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.16).nonzero()
                    idx1 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.16) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.49)).nonzero()
                    idx2 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.49) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.82)).nonzero()
                    idx3 = (attr_gt[batchidx_with_attri, capsule_idx] >= 0.82).nonzero()
                    idxs = [idx0, idx1, idx2, idx3]
                    L_cluster = 0

                    for idxi in range(len(idxs)):
                        if len(idxs[idxi]) > 0:
                            L_cluster += torch.sum(
                                torch.min(torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]),dim=-1)[
                                    0])

                    L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                    L_sep_loss = sep_loss(max_dist=max_dist,
                                          selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],
                                          indices=[idx0, idx1, idx2, idx3])

                    L_sep += L_sep_loss / (len(batchidx_with_attri) * 3)

                elif capsule_idx == 2:
                    idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.1).nonzero()
                    idx1 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.1) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.3)).nonzero()
                    idx2 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.3) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.5)).nonzero()
                    idx3 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.5) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.7)).nonzero()
                    idx4 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.7) & (
                            attr_gt[batchidx_with_attri, capsule_idx] < 0.9)).nonzero()
                    idx5 = (attr_gt[batchidx_with_attri, capsule_idx] >= 0.9).nonzero()
                    idxs = [idx0, idx1, idx2, idx3, idx4, idx5]
                    L_cluster = 0

                    for idxi in range(len(idxs)):
                        if len(idxs[idxi]) > 0:
                            L_cluster += torch.sum(
                                torch.min(torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]),dim=-1)[
                                    0])

                    L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                    L_sep_loss = sep_loss(max_dist=max_dist,
                                          selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],
                                          indices=[idx0, idx1, idx2, idx3, idx4, idx5])

                    L_sep += L_sep_loss / (len(batchidx_with_attri) * 5)

        L_sep /= len(dists_to_protos)
        L_cluster_allmean = L_cluster_allcpsi / len(dists_to_protos)
        total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr + (1 / 8) * (
                L_cluster_allmean + 0.1 * L_sep)
    return total_loss, L_pred, L_recon, L_attr, L_cluster_allmean, L_sep


def sep_loss(max_dist, selected_dists, indices):
        num_classes = len(indices)
        ranges = [list(range(1,num_classes))]
        for i in range(num_classes-2):
            ranges.append(list(range(0,num_classes-(num_classes-1-i)))+list(range(i+2,num_classes)))
        ranges.append(list(range(0,num_classes-1)))
        loss_temp = []
        for i in range(num_classes):
            loss_temp.append(torch.min(torch.maximum(torch.zeros_like(
                max_dist - selected_dists[indices[i],ranges[i]]),
                (max_dist - selected_dists[indices[i],ranges[i]])),
                dim=-1)[0])
        return torch.sum(torch.cat(loss_temp, dim=0))


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

    class Args:
        warmup = 1

    print("=" * 70)
    print("Proto-Caps core model component benchmark")
    print("=" * 70)

    # ==========================================================
    # Test 1: squash and PrimaryCapsule
    # ==========================================================
    print("[Test 1/6] capsule squash and primary capsules")
    try:
        x = torch.randn(2, 5, 8)
        y = squash(x)
        primary = PrimaryCapsule(256, 256, 8, kernel_size=9, threeD=False, stride=2, padding=0)
        conv_input = torch.randn(2, 256, 24, 24)
        caps = primary(conv_input)
        check("squash output not None", y is not None)
        check("squash shape preserved", tuple(y.shape) == (2, 5, 8), f"got {tuple(y.shape)}")
        check("squash finite", torch.isfinite(y).all().item())
        check("squash vector norms bounded", (torch.norm(y, dim=-1) <= 1.0 + 1e-5).all().item())
        check("primary output shape", tuple(caps.shape) == (2, 2048, 8), f"got {tuple(caps.shape)}")
        check("primary capsule norms bounded", (torch.norm(caps, dim=-1) <= 1.0 + 1e-5).all().item())
    except Exception as exc:
        skip_checks(6, f"squash/PrimaryCapsule raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2: DenseCapsule dynamic routing
    # ==========================================================
    print("[Test 2/6] dense capsule dynamic routing")
    try:
        if not torch.cuda.is_available():
            skip_checks(6, "source DenseCapsule initializes routing logits on CUDA")
        else:
            dense = DenseCapsule(in_num_caps=6, in_dim_caps=4, out_num_caps=3, out_dim_caps=5, routings=3, activation_fn="softmax").cuda()
            x = torch.randn(2, 6, 4, device="cuda", requires_grad=True)
            out = dense(x)
            check("dense capsule output not None", out is not None)
            check("dense capsule output shape", tuple(out.shape) == (2, 3, 5), f"got {tuple(out.shape)}")
            check("dense capsule output finite", torch.isfinite(out).all().item())
            check("dense capsule norms bounded", (torch.norm(out, dim=-1) <= 1.0 + 1e-5).all().item())
            out.sum().backward()
            check("dense capsule input grad exists", x.grad is not None)
            check("dense capsule weight grad exists", dense.weight.grad is not None)
    except Exception as exc:
        skip_checks(6, f"DenseCapsule raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3: ProtoCapsNet.forwardCapsule and getDistance
    # ==========================================================
    print("[Test 3/6] ProtoCapsNet capsule heads and prototype distances")
    try:
        model = ProtoCapsNet(input_size=[1, 32, 32], numcaps=8, routings=3,
                             out_dim_caps=16, activation_fn="sigmoid", threeD=False, numProtos=2)
        x_caps = torch.randn(2, 8, 16)
        pred, pred_attr, recon = model.forwardCapsule(x_caps)
        dists = model.getDistance(x_caps)
        check("forwardCapsule prediction shape", tuple(pred.shape) == (2, 5), f"got {tuple(pred.shape)}")
        check("forwardCapsule attr shape", tuple(pred_attr.shape) == (2, 8), f"got {tuple(pred_attr.shape)}")
        check("forwardCapsule recon shape", tuple(recon.shape) == (2, 1, 32, 32), f"got {tuple(recon.shape)}")
        check("prediction probabilities sum to one", torch.allclose(pred.sum(dim=1), torch.ones(2), atol=1e-5))
        check("distance list length", len(dists) == 8)
        check("distance capsule0 shape", tuple(dists[0].shape) == (2, 5, 2), f"got {tuple(dists[0].shape)}")
        check("distances nonnegative", all((d >= 0).all().item() for d in dists))
    except Exception as exc:
        skip_checks(7, f"ProtoCapsNet heads/distances raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4: ProtoCapsNet.forward
    # ==========================================================
    print("[Test 4/6] ProtoCapsNet end-to-end forward")
    try:
        if not torch.cuda.is_available():
            skip_checks(7, "source DenseCapsule inside ProtoCapsNet initializes routing logits on CUDA")
        else:
            model = ProtoCapsNet(input_size=[1, 32, 32], numcaps=8, routings=2,
                                 out_dim_caps=16, activation_fn="sigmoid", threeD=False, numProtos=2).cuda()
            image = torch.randn(2, 1, 32, 32, device="cuda")
            pred, pred_attr, recon, dists = model(image)
            check("ProtoCaps pred shape", tuple(pred.shape) == (2, 5), f"got {tuple(pred.shape)}")
            check("ProtoCaps attr shape", tuple(pred_attr.shape) == (2, 8), f"got {tuple(pred_attr.shape)}")
            check("ProtoCaps recon shape", tuple(recon.shape) == (2, 1, 32, 32), f"got {tuple(recon.shape)}")
            check("ProtoCaps pred finite", torch.isfinite(pred).all().item())
            check("ProtoCaps pred normalized", torch.allclose(pred.sum(dim=1), torch.ones(2, device="cuda"), atol=1e-5))
            check("ProtoCaps distance count", len(dists) == 8)
            check("ProtoCaps distance capsule2 class count", dists[2].shape[1] == 6)
    except Exception as exc:
        skip_checks(7, f"ProtoCapsNet forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5: sep_loss
    # ==========================================================
    print("[Test 5/6] prototype separation loss")
    try:
        selected = torch.rand(3, 5, 2)
        indices = [
            torch.tensor([[0]]),
            torch.tensor([[1]]),
            torch.tensor([[2]]),
            torch.empty(0, 1, dtype=torch.long),
            torch.empty(0, 1, dtype=torch.long),
        ]
        loss = sep_loss(max_dist=16, selected_dists=selected, indices=indices)
        check("sep loss not None", loss is not None)
        check("sep loss scalar", loss.dim() == 0)
        check("sep loss finite", torch.isfinite(loss).item())
        check("sep loss nonnegative", loss.item() >= 0)
    except Exception as exc:
        skip_checks(4, f"sep_loss raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 6: xcaps_loss warmup and prototype branch
    # ==========================================================
    print("[Test 6/6] combined Proto-Caps loss")
    try:
        batch = 3
        y = F.softmax(torch.randn(batch, 5), dim=1)
        y_pred = F.softmax(torch.randn(batch, 5), dim=1)
        x = torch.rand(batch, 1, 32, 32)
        x_recon = torch.rand(batch, 1, 32, 32)
        attr_gt = torch.rand(batch, 8)
        attr_pred = torch.rand(batch, 8)
        sample_id = [10, 11, 12]
        idx_with_attri = [10, 11, 12]
        dists = []
        class_counts = [5, 4, 6, 5, 5, 5, 5, 5]
        for count in class_counts:
            dists.append(torch.rand(batch, count, 2))
        warm = xcaps_loss(y, y_pred, x, x_recon, 0.5, attr_gt, attr_pred, 0.25,
                          epoch=0, dists_to_protos=dists, sample_id=sample_id,
                          idx_with_attri=idx_with_attri, max_dist=16, arguments=Args())
        full = xcaps_loss(y, y_pred, x, x_recon, 0.5, attr_gt, attr_pred, 0.25,
                          epoch=2, dists_to_protos=dists, sample_id=sample_id,
                          idx_with_attri=idx_with_attri, max_dist=16, arguments=Args())
        check("warmup loss tuple length", len(warm) == 6)
        check("full loss tuple length", len(full) == 6)
        check("warmup total scalar", warm[0].dim() == 0)
        check("full total scalar", full[0].dim() == 0)
        check("loss values finite", all(torch.isfinite(v).item() if torch.is_tensor(v) else True for v in full))
        check("prototype cluster populated after warmup", torch.is_tensor(full[4]))
    except Exception as exc:
        skip_checks(6, f"xcaps_loss raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some target functions may be incomplete.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
