# ============================================================
# ground_truth.py - RPNODE_FSS Core Model Components
# Source: Adversarial/RPNODE_FSS-main
#
# Contains ONLY the VGG encoder, prototypical few-shot segmenter,
# and regularized Neural-ODE feature module.
# ============================================================

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


# --- [Original file: models/vgg.py] ---

class Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """
    def __init__(self, in_channels=3, pretrained_path=None, rem_last_layer=False, pretrained_ode=False, last_2_layers = 3):
        super().__init__()
        self.pretrained_path = pretrained_path

        if rem_last_layer:
            self.features = nn.Sequential(
                self._make_layer(2, in_channels, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._make_layer(2, 64, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._make_layer(3, 128, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._make_layer(last_2_layers, 256, 512),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                # self._make_layer(3, 512, 512, dilation=2, lastRelu=False),
            )
        else:
            self.features = nn.Sequential(
                self._make_layer(2, in_channels, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._make_layer(2, 64, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._make_layer(3, 128, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._make_layer(3, 256, 512),
                # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                self._make_layer(3, 512, 512, dilation=2, lastRelu=False),
            )
        self.pretrained_ode = pretrained_ode

        self._init_weights()

    def forward(self, x):
        return self.features(x)

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if self.pretrained_path is not None:
            dic = torch.load(self.pretrained_path, map_location='cpu')
            keys = list(dic.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(4,26):
                try:
                    new_dic[new_keys[i]] = dic[keys[i]]
                except IndexError:
                    if not self.pretrained_ode:
                        print("VGG model weight {} not initialised".format(i))
                        continue
                    elif self.pretrained_ode and i<20:
                        print("VGG model weight {} not initialised".format(i))
                        continue

            self.load_state_dict(new_dic)


# --- [Original file: models/fewshot.py] ---

class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=1, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)),]))


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, factor=1, return_feats=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 1 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 1 x H x W], list of tensors
        """
        self.factor = factor
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H x W
        # back_mask = torch.stack([torch.stack(way, dim=0)
        #                          for way in back_mask], dim=0)  # Wa x Sh x B x H x W

        back_mask = torch.ones_like(fore_mask) - fore_mask
        ###### Compute loss ######
        outputs = []
        all_prototypes = []
        all_fg_prototypess = []
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            all_prototypes += prototypes
            all_fg_prototypess += fg_prototypes

            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

        all_prototypes = torch.stack(all_prototypes,  dim=0)
        all_fg_prototypess = torch.stack(all_fg_prototypess,  dim=0)
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        if return_feats:
            return output, all_prototypes, qry_fts
        return output, all_fg_prototypess

    def get_sup_fore(self, sup, fore_mask):
        # print(len(sup), len(sup[0]), len(sup[0][0]), len(sup[0][0][0]), len(sup[0][0][0][0]))
        batch_size = sup[0].shape[0]
        # print("batch size: ", batch_size)
        # sup = torch.cat([torch.cat(sample, dim=0) for sample in sup])
        sup = torch.cat(sup, dim=0)
        sup = sup.squeeze(1)
        # print(sup.shape)
        supp_fts = self.encoder(sup)
        # print(len(fore_mask), len(fore_mask[0]), len(fore_mask[0][0]), fore_mask[0][0][0].shape)
        num_samples = (supp_fts.shape[0]//batch_size)
        # print(num_samples)
        # print(supp_fts.shape, fore_mask.shape)
        supp_fg_fts = [self.getFeatures(supp_fts[i, ...].unsqueeze(0),  fore_mask[i%num_samples][0][i//num_samples].unsqueeze(0)) for i in range(supp_fts.shape[0])]
        supp_fg_fts = [torch.cat(supp_fg_fts[i*batch_size: (i+1)*batch_size], dim=0) for i in range(len(supp_fg_fts)//batch_size)]
        return supp_fg_fts


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        # print(fts.shape, mask.shape)
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        if n_shots > self.factor:
            n_shots = n_shots // self.factor
        fg_prototypes = [sum(way[:n_shots]) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way[:n_shots]) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype


# --- [Original file: models/ode.py] ---

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)



class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim, n_layers=3, sigma=0.1, noise_type="additive"):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.layers = []
        self.norm = norm(dim)
        for i in range(n_layers):
            self.layers.append(
                {
                    "conv": ConcatConv2d(dim, dim, 3, 1, 1),
                    "norm": norm(dim),
                }
            )
        lyrs = [l["conv"] for l in self.layers] + [l["norm"] for l in self.layers]
        self.layers_seq = torch.nn.Sequential(*lyrs)
        self.nfe = 0
        self.sigma = sigma
        self.noise_type = noise_type

    def forward(self, t, x):
        self.nfe += 1

        out = self.norm1(x)
        for i in range(len(self.layers)):
            out = self.relu(out)
            out = self.layers[i]["conv"](t, out)
            out = self.layers[i]["norm"](out)

        return out




class ODEBlock(nn.Module):

    def __init__(self, odefunc, ode_time=1):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, ode_time]).float()
        self.tol = 1e-3

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value




class ODENet(nn.Module):
    def __init__(self, in_channels, pretrained_path=None, ode_layers=3, ode_time=1, noise_type=None, sigma=None):
        super(ODENet, self).__init__()
        self.ode = ODEBlock(ODEfunc(in_channels, n_layers=ode_layers, noise_type=noise_type, sigma=sigma), ode_time=ode_time)

    def forward(self, x):
        return self.ode(x)



class FewShotSegOde(FewShotSeg):
    def __init__(self, in_channels=1, pretrained_path=None, pretrained_ode=False, ode_layers=3, ode_time=1, noise_type="None", sigma=None):
        super().__init__(in_channels=in_channels, pretrained_path=pretrained_path)
        ode_weights = pretrained_path if pretrained_ode else None
        # Encoder
        if ode_layers == 5:
            last_2_layers = 1
        elif ode_layers == 4:
            last_2_layers = 2
        else:
            last_2_layers = 3
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ('backbone', Encoder(in_channels, self.pretrained_path, rem_last_layer=True, pretrained_ode=pretrained_ode, last_2_layers=last_2_layers)),
                    ('ode', ODENet(512, pretrained_path=ode_weights, ode_layers=ode_layers, ode_time=ode_time, noise_type=noise_type, sigma=sigma)),
                ]
            )
        )


# ============================================================
# __main__: Automated test suite for 7 ablated functions
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

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                self.conv.weight.copy_(torch.tensor([[[[1.0]]], [[[0.5]]], [[[-1.0]]]]))

        def forward(self, x):
            return self.conv(x)

    def make_tiny_segmenter():
        model = FewShotSeg.__new__(FewShotSeg)
        nn.Module.__init__(model)
        model.pretrained_path = None
        model.encoder = TinyEncoder()
        return model

    print("=" * 70)
    print("RPNODE_FSS Benchmark: Prototypical Segmentation with Neural ODE")
    print("Automated Test Suite - 7 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/7: ConcatConv2d.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 1/7] ConcatConv2d.forward - append integration time as a feature channel")
    try:
        layer = ConcatConv2d(2, 3, ksize=3, padding=1).to(device)
        x = torch.randn(4, 2, 5, 5, device=device, requires_grad=True)
        y = layer(torch.tensor(0.25, device=device), x)
        check("ConcatConv2d output not None", y is not None)
        if y is not None:
            check("ConcatConv2d output shape", y.shape == (4, 3, 5, 5), f"expected (4, 3, 5, 5), got {tuple(y.shape)}")
            check("ConcatConv2d output finite", torch.isfinite(y).all().item())
            check("ConcatConv2d time channel parameterized", layer._layer.weight.shape[1] == 3)
            y.sum().backward()
            check("ConcatConv2d input gradient", x.grad is not None and torch.isfinite(x.grad).all().item())
            check("ConcatConv2d weight gradient", layer._layer.weight.grad is not None and torch.isfinite(layer._layer.weight.grad).all().item())
        else:
            skip_checks(5, "ConcatConv2d.forward returned None")
    except Exception as exc:
        skip_checks(6, f"ConcatConv2d.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/7: ODEfunc.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 2/7] ODEfunc.forward - residual dynamics right-hand side")
    try:
        func = ODEfunc(4, n_layers=2).to(device)
        x = torch.randn(2, 4, 6, 6, device=device, requires_grad=True)
        before_nfe = func.nfe
        y = func(torch.tensor(0.5, device=device), x)
        check("ODEfunc output not None", y is not None)
        if y is not None:
            check("ODEfunc output shape", y.shape == (2, 4, 6, 6), f"expected (2, 4, 6, 6), got {tuple(y.shape)}")
            check("ODEfunc output finite", torch.isfinite(y).all().item())
            check("ODEfunc nfe increments", func.nfe == before_nfe + 1, f"expected {before_nfe + 1}, got {func.nfe}")
            y.sum().backward()
            check("ODEfunc input gradient", x.grad is not None and torch.isfinite(x.grad).all().item())
            conv_grads = [layer["conv"]._layer.weight.grad for layer in func.layers]
            check("ODEfunc conv gradients", all(g is not None and torch.isfinite(g).all().item() for g in conv_grads))
        else:
            skip_checks(5, "ODEfunc.forward returned None")
    except Exception as exc:
        skip_checks(6, f"ODEfunc.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/7: ODEBlock.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 3/7] ODEBlock.forward - integrate feature dynamics")
    try:
        block = ODEBlock(ODEfunc(4, n_layers=1), ode_time=0.1).to(device)
        x = torch.randn(2, 4, 5, 5, device=device, requires_grad=True)
        y = block(x)
        check("ODEBlock output not None", y is not None)
        if y is not None:
            check("ODEBlock output shape", y.shape == (2, 4, 5, 5), f"expected (2, 4, 5, 5), got {tuple(y.shape)}")
            check("ODEBlock output finite", torch.isfinite(y).all().item())
            check("ODEBlock integration time dtype", block.integration_time.dtype == x.dtype)
            check("ODEBlock nfe positive", block.nfe > 0)
            y.sum().backward()
            check("ODEBlock input gradient", x.grad is not None and torch.isfinite(x.grad).all().item())
        else:
            skip_checks(5, "ODEBlock.forward returned None")
    except Exception as exc:
        skip_checks(6, f"ODEBlock.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/7: FewShotSeg.getFeatures
    # ==========================================================
    print("-" * 60)
    print("[Test 4/7] FewShotSeg.getFeatures - masked average prototype features")
    try:
        model = make_tiny_segmenter().to(device)
        fts = torch.arange(48, dtype=torch.float32, device=device).view(1, 3, 4, 4).requires_grad_(True)
        mask = torch.zeros(1, 4, 4, device=device)
        mask[:, :2, :2] = 1
        feat = model.getFeatures(fts, mask)
        expected = fts[:, :, :2, :2].sum(dim=(2, 3)) / (mask.sum(dim=(1, 2), keepdim=False).view(1, 1) + 1e-5)
        check("getFeatures output not None", feat is not None)
        if feat is not None:
            check("getFeatures output shape", feat.shape == (1, 3), f"expected (1, 3), got {tuple(feat.shape)}")
            check("getFeatures output finite", torch.isfinite(feat).all().item())
            check("getFeatures masked mean", torch.allclose(feat, expected, atol=1e-5))
            feat.sum().backward()
            check("getFeatures selected gradient", fts.grad[:, :, :2, :2].abs().sum().item() > 0)
            check("getFeatures masked gradient zero", fts.grad[:, :, 2:, 2:].abs().sum().item() == 0)
        else:
            skip_checks(5, "FewShotSeg.getFeatures returned None")
    except Exception as exc:
        skip_checks(6, f"FewShotSeg.getFeatures raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5/7: FewShotSeg.getPrototype
    # ==========================================================
    print("-" * 60)
    print("[Test 5/7] FewShotSeg.getPrototype - foreground/background prototype averaging")
    try:
        model = make_tiny_segmenter().to(device)
        model.factor = 2
        fg_fts = [
            [torch.tensor([[1.0, 2.0, 3.0]], device=device), torch.tensor([[3.0, 4.0, 5.0]], device=device), torch.tensor([[5.0, 6.0, 7.0]], device=device)],
            [torch.tensor([[2.0, 0.0, 1.0]], device=device), torch.tensor([[4.0, 2.0, 3.0]], device=device), torch.tensor([[6.0, 4.0, 5.0]], device=device)],
        ]
        bg_fts = [
            [torch.tensor([[0.0, 1.0, 2.0]], device=device), torch.tensor([[2.0, 3.0, 4.0]], device=device), torch.tensor([[4.0, 5.0, 6.0]], device=device)],
            [torch.tensor([[1.0, 1.0, 1.0]], device=device), torch.tensor([[3.0, 3.0, 3.0]], device=device), torch.tensor([[5.0, 5.0, 5.0]], device=device)],
        ]
        fg_prototypes, bg_prototype = model.getPrototype(fg_fts, bg_fts)
        check("getPrototype fg not None", fg_prototypes is not None)
        check("getPrototype bg not None", bg_prototype is not None)
        if fg_prototypes is not None and bg_prototype is not None:
            check("getPrototype fg count", len(fg_prototypes) == 2, f"expected 2, got {len(fg_prototypes)}")
            check("getPrototype fg shape", all(proto.shape == (1, 3) for proto in fg_prototypes))
            check("getPrototype bg shape", bg_prototype.shape == (1, 3), f"expected (1, 3), got {tuple(bg_prototype.shape)}")
            check("getPrototype shot truncation fg", torch.allclose(fg_prototypes[0], fg_fts[0][0]))
            expected_bg = (bg_fts[0][0] + bg_fts[1][0]) / 2
            check("getPrototype background average", torch.allclose(bg_prototype, expected_bg))
        else:
            skip_checks(5, "FewShotSeg.getPrototype returned None")
    except Exception as exc:
        skip_checks(7, f"FewShotSeg.getPrototype raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 6/7: FewShotSeg.calDist
    # ==========================================================
    print("-" * 60)
    print("[Test 6/7] FewShotSeg.calDist - cosine prototype distance map")
    try:
        model = make_tiny_segmenter().to(device)
        prototype = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        fts = torch.zeros(2, 3, 4, 4, device=device)
        fts[:, 0, :, :] = 1.0
        fts.requires_grad_(True)
        dist = model.calDist(fts, prototype, scaler=20)
        check("calDist output not None", dist is not None)
        if dist is not None:
            check("calDist output shape", dist.shape == (2, 4, 4), f"expected (2, 4, 4), got {tuple(dist.shape)}")
            check("calDist output finite", torch.isfinite(dist).all().item())
            check("calDist cosine scale", torch.allclose(dist, torch.full((2, 4, 4), 20.0, device=device), atol=1e-5))
            dist.sum().backward()
            check("calDist input gradient", fts.grad is not None and torch.isfinite(fts.grad).all().item())
        else:
            skip_checks(4, "FewShotSeg.calDist returned None")
    except Exception as exc:
        skip_checks(5, f"FewShotSeg.calDist raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 7/7: FewShotSeg.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 7/7] FewShotSeg.forward - end-to-end prototypical segmentation episode")
    try:
        model = make_tiny_segmenter().to(device)
        support = torch.randn(2, 1, 8, 8, device=device, requires_grad=True)
        query = torch.randn(2, 1, 8, 8, device=device, requires_grad=True)
        fore = torch.zeros(2, 8, 8, device=device)
        fore[:, :4, :4] = 1
        supp_imgs = [[support]]
        fore_mask = [[fore]]
        back_mask = [[1 - fore]]
        qry_imgs = [query]
        output, fg_proto = model(supp_imgs, fore_mask, back_mask, qry_imgs, factor=1, return_feats=False)
        check("FewShotSeg forward output not None", output is not None)
        check("FewShotSeg forward prototype not None", fg_proto is not None)
        if output is not None and fg_proto is not None:
            check("FewShotSeg output shape", output.shape == (2, 2, 8, 8), f"expected (2, 2, 8, 8), got {tuple(output.shape)}")
            check("FewShotSeg foreground prototype shape", fg_proto.shape == (2, 1, 3), f"expected (2, 1, 3), got {tuple(fg_proto.shape)}")
            check("FewShotSeg output finite", torch.isfinite(output).all().item())
            check("FewShotSeg class channels", output.shape[1] == 2)
            output.sum().backward()
            check("FewShotSeg query gradient", query.grad is not None and torch.isfinite(query.grad).all().item())
            check("FewShotSeg support gradient", support.grad is not None and torch.isfinite(support.grad).all().item())
        else:
            skip_checks(6, "FewShotSeg.forward returned None for output or prototype")
    except Exception as exc:
        skip_checks(8, f"FewShotSeg.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Final Score
    # ==========================================================
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
