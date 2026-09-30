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
    [TODO] Apply capsule vector squashing along one capsule dimension.

    Input:
        inputs: (..., dim_caps, ...) - capsule vectors arranged along `axis`.
        axis: int - the capsule-vector dimension whose length should be normalized.

    Output: same shape as `inputs` - vectors with preserved directions and bounded lengths.

"""
    pass


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
        """
        [TODO] Transform input capsules into output capsules using dynamic routing.

        Input:
            x: (batch, in_num_caps, in_dim_caps) - capsule vectors from the previous layer.

        Output: (batch, out_num_caps, out_dim_caps) - routed output capsule vectors.

"""
        pass


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
        """
        [TODO] Convert a convolutional feature map into primary capsule vectors.

        Input:
            x: (batch, channels, height, width) for 2D mode, or
               (batch, channels, depth, height, width) for 3D mode.

        Output: (batch, num_primary_caps, dim_caps) - flattened primary capsule vectors.

"""
        pass


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
        """
        [TODO] Decode already-computed attribute capsules into prediction, attributes, and reconstruction.

        Input:
            x_ex: (batch, 8, out_dim_caps) - one capsule vector per semantic attribute.

        Output:
            pred: (batch, 5) - class probability vector.
            pred_attr: (batch, 8) - scalar attribute predictions.
            reconstruction: (batch, *input_size) - reconstructed input-shaped tensor.

"""
        pass

    def forward(self, x):
        """
        [TODO] Run the complete Proto-Caps forward path from image tensor to predictions and prototype distances.

        Input:
            x: (batch, *input_size) - 2D or 3D medical image tensor configured at construction time.

        Output:
            pred: (batch, 5) - class probability vector.
            pred_attr: (batch, 8) - scalar attribute predictions.
            reconstruction: (batch, *input_size) - decoder reconstruction.
            dists_to_protos: list of 8 tensors - capsule-wise distances to learned prototypes.

"""
        pass

    def getDistance(self, x):
        """
        [TODO] Compute capsule-wise distances from semantic capsules to prototype vectors.

        Input:
            x: (batch, 8, out_dim_caps) - routed semantic capsule vectors.

        Output:
            list of 8 tensors:
                capsule 0: (batch, 1, 5, numProtos)
                capsule 1: (batch, 1, 4, numProtos)
                capsule 2: (batch, 1, 6, numProtos)
                capsules 3-7: (batch, 1, 5, numProtos)

"""
        pass


# --- [Original file: loss.py] ---
def xcaps_loss(y, y_pred, x, x_recon, lam_recon, attr_gt, attr_pred, lam_attr, epoch,
               dists_to_protos, sample_id, idx_with_attri, max_dist, arguments):
    """
    [TODO] Compute Proto-Caps prediction, reconstruction, attribute, cluster, and separation losses.

    Input:
        y: (batch, 5) - target class distribution.
        y_pred: (batch, 5) - predicted class distribution.
        x: (batch, *input_size) - original input tensor.
        x_recon: (batch, *input_size) - reconstructed input tensor.
        lam_recon: scalar - reconstruction-loss weight.
        attr_gt: (batch, 8) - continuous attribute targets.
        attr_pred: (batch, 8) - predicted attributes.
        lam_attr: scalar - attribute-loss weight.
        epoch: int - current epoch controlling warmup behavior.
        dists_to_protos: list of 8 tensors - prototype distances per capsule.
        sample_id: sequence of length batch - identifiers for samples in the batch.
        idx_with_attri: sequence - identifiers whose attribute labels are available.
        max_dist: scalar - margin used by prototype separation.
        arguments: object - contains the warmup epoch threshold.

    Output:
        tuple of 6 scalar losses:
            total loss, prediction loss, reconstruction loss, attribute loss,
            mean prototype-cluster loss, prototype-separation loss.

"""
    pass


def sep_loss(max_dist, selected_dists, indices):
    """
    [TODO] Penalize samples that are too close to prototypes from other attribute bins.

    Input:
        max_dist: scalar - separation margin.
        selected_dists: (annotated_batch, 1, num_classes, numProtos) - distances for one capsule.
        indices: list of length num_classes - sample indices belonging to each attribute bin.

    Output: scalar tensor - summed margin penalty over cross-bin prototype distances.

"""
    pass


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
