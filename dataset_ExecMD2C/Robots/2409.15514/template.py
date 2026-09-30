import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.transforms import v2
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import lightning.pytorch as pl
from torch_geometric.nn.models import GraphSAGE
from pytorch_metric_learning import losses
from sklearn.metrics.pairwise import haversine_distances
torchvision.disable_beta_transforms_warning()


class ConvNextExtractor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.map_conv = torch.load(f'{self.args.path}/pretrained/convnextv2_tiny_22k_384_ema.pt', map_location=self.device)
        self.map_conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.map_conv.classifier[2] = nn.Identity()

        # self.pov_conv = torch.load(f'{self.args.path}/pretrained/convnextv2_tiny_22k_384_ema.pt', map_location=self.device)
        self.pov_conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.pov_conv.classifier[2] = nn.Identity()

    def embed_map(self, map_tile: torch.Tensor) -> torch.Tensor:
        """
        [TODO] Embed satellite/map image tiles with the map ConvNeXt branch.

        Input:
            map_tile: (num_nodes, 3, height, width) - normalized satellite image tensor.

        Output:
            (num_nodes, 768) - one image descriptor per graph node.

"""
        pass

    def embed_pov(self, pov_tile: torch.Tensor):
        """
        [TODO] Embed street-view/point-of-view image tiles with the POV ConvNeXt branch.

        Input:
            pov_tile: (num_nodes, 3, height, width) - normalized panorama crop tensor.

        Output:
            (num_nodes, 768) - one image descriptor per graph node.

"""
        pass


class FullModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.feat_extractor = ConvNextExtractor()
        self.encoder = GraphSAGE(in_channels=768, hidden_channels=256, num_layers=2, out_channels=64)

        self.augmentor = v2.Compose([#v2.RandomResizedCrop(size=(224, 224), antialias=True), # v2.RandomHorizontalFlip(p=0.1), # Only add these once models are working
                                    v2.ToDtype(torch.float32), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.val_process = v2.Compose([#v2.Resize(size=(224, 224), antialias=True),
                                    v2.ToDtype(torch.float32), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.loss_function = losses.TripletMarginLoss(margin=0.1) # NTXent?

        self.batch_size = self.hparams['args'].batch_size
        self.lr = self.hparams['args'].lr

        self.current_val_loss = 1000000

        self.train_loss, self.val_loss = [], []
        self.train_a, self.train_b, self.val_a, self.val_b = [], [], [], []
        self.train_pointers, self.val_pointers, self.test_pointers = [], [], []
        self.level_of_distance = -1
        self.gt_ori_train, self.est_ori_train = [], []
        self.gt_ori_val, self.est_ori_val = [], []

        self.test_loss = []
        self.test_a, self.test_b = [], []
        self.gt_ori_test, self.est_ori_test = [], []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, img='sat') -> torch.Tensor:
        """
        [TODO] Encode one graph-view image batch into normalized node embeddings.

        Input:
            x: (num_nodes, 3, height, width) - satellite or POV image batch.
            edge_index: (2, num_edges) - graph connectivity for the node sequence.
            img: string - either 'sat' for satellite/map or 'pov' for street-view images.

        Output:
            (num_nodes, 64) - L2-normalized graph-aware node embeddings.

"""
        pass

    def triplet_mining(self, batch, z_a, z_b):
        """
        [TODO] Construct triplet-loss indices for paired POV and satellite graph embeddings.

        Input:
            batch: graph batch with 'pos' coordinates and 'ptr' graph-start pointers.
            z_a: (num_nodes, embedding_dim) - POV embeddings.
            z_b: (num_nodes, embedding_dim) - satellite/map embeddings.

        Output:
            embeddings: (2 * num_nodes, embedding_dim) - concatenated POV then map embeddings.
            anchors: (num_nodes,) - indices into the POV half.
            positives: (num_nodes,) - matching indices into the satellite half.
            negatives: (num_nodes,) - non-matching satellite indices.

"""
        pass

    def walk_step(self, batch, batch_idx, stage='train'):
        """
        [TODO] Run one SpaGBOL graph-walk step for train, validation, or test.

        Input:
            batch: graph batch containing sat_image, pov_image, edge_index, ptr, pos, and yaws_image.
            batch_idx: integer batch index, retained for Lightning compatibility.
            stage: 'train', 'val', or 'test' - controls preprocessing and metric buffers.

        Output:
            scalar tensor - triplet metric-learning loss for the batch.

"""
        pass

    def training_step(self, batch, batch_idx):
        loss = self.walk_step(batch=batch, batch_idx=batch_idx, stage='train')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.walk_step(batch=batch, batch_idx=batch_idx, stage='val')
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self.walk_step(batch=batch, batch_idx=batch_idx, stage='test')
        return {'loss': loss}


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
    print("SpaGBOL: graph-based cross-view localisation benchmark")
    print("=" * 70)

    class FakeConvNext(nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset
            self.classifier = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Linear(768, 768)])

        def forward(self, x):
            pooled = x.mean(dim=(2, 3))
            repeat_count = 768 // pooled.shape[1]
            return pooled.repeat(1, repeat_count) + self.offset

    try:
        conv_calls = {"count": 0}
        original_convnext_tiny = globals()["convnext_tiny"]

        def fake_convnext_tiny(weights=None):
            conv_calls["count"] += 1
            return FakeConvNext(offset=float(conv_calls["count"]))

        globals()["convnext_tiny"] = fake_convnext_tiny
        extractor = ConvNextExtractor()
        image = torch.ones(2, 3, 8, 8)
        map_emb = extractor.embed_map(image)
        pov_emb = extractor.embed_pov(image)
        check("ConvNextExtractor map output not None", map_emb is not None)
        if map_emb is not None:
            check("ConvNextExtractor map shape", tuple(map_emb.shape) == (2, 768), f"got {tuple(map_emb.shape)}")
            check("ConvNextExtractor map finite", torch.isfinite(map_emb).all().item())
            check("ConvNextExtractor strips map classifier head", isinstance(extractor.map_conv.classifier[2], nn.Identity))
        else:
            skip_checks(3, "map embedding returned None")
        check("ConvNextExtractor pov output not None", pov_emb is not None)
        if pov_emb is not None:
            check("ConvNextExtractor pov shape", tuple(pov_emb.shape) == (2, 768), f"got {tuple(pov_emb.shape)}")
            check("ConvNextExtractor pov finite", torch.isfinite(pov_emb).all().item())
            check("ConvNextExtractor keeps separate map and pov branches", not torch.allclose(map_emb, pov_emb))
        else:
            skip_checks(3, "pov embedding returned None")
        globals()["convnext_tiny"] = original_convnext_tiny
    except Exception as exc:
        if "original_convnext_tiny" in locals():
            globals()["convnext_tiny"] = original_convnext_tiny
        skip_checks(8, f"ConvNextExtractor raised {exc!r}")

    try:
        class BranchExtractor:
            def embed_map(self, map_tile):
                return torch.ones(map_tile.shape[0], 768) * 2.0

            def embed_pov(self, pov_tile):
                return torch.ones(pov_tile.shape[0], 768) * -3.0

        class FakeEncoder:
            def __call__(self, x, edge_index):
                idx = torch.arange(64, dtype=x.dtype).unsqueeze(0).repeat(x.shape[0], 1)
                return idx + x[:, :1]

        model = types.SimpleNamespace(feat_extractor=BranchExtractor(), encoder=FakeEncoder())
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        sat_out = FullModel.forward(model, torch.randn(4, 3, 8, 8), edge_index=edge_index, img='sat')
        pov_out = FullModel.forward(model, torch.randn(4, 3, 8, 8), edge_index=edge_index, img='pov')
        check("FullModel.forward sat output not None", sat_out is not None)
        if sat_out is not None:
            check("FullModel.forward sat shape", tuple(sat_out.shape) == (4, 64), f"got {tuple(sat_out.shape)}")
            check("FullModel.forward sat finite", torch.isfinite(sat_out).all().item())
            check("FullModel.forward L2 normalizes rows", torch.allclose(sat_out.norm(dim=1), torch.ones(4), atol=1e-6))
            check("FullModel.forward separates sat and pov branches", not torch.allclose(sat_out, pov_out))
        else:
            skip_checks(4, "forward returned None")
    except Exception as exc:
        skip_checks(5, f"FullModel.forward raised {exc!r}")

    try:
        args = types.SimpleNamespace(triplet_mine=False, walk=2)
        model = types.SimpleNamespace(hparams={"args": args}, level_of_distance=-1)
        z_a = F.normalize(torch.randn(6, 64), dim=1)
        z_b = F.normalize(torch.randn(6, 64), dim=1)
        batch = {"pos": torch.randn(6, 2), "ptr": torch.tensor([0, 2, 4, 6])}
        embeddings, anchors, positives, negatives = FullModel.triplet_mining(model, batch, z_a, z_b)
        check("triplet_mining random output not None", embeddings is not None)
        if embeddings is not None:
            check("triplet_mining random embeddings shape", tuple(embeddings.shape) == (12, 64), f"got {tuple(embeddings.shape)}")
            check("triplet_mining random anchors shape", tuple(anchors.shape) == (6,))
            check("triplet_mining random positives offset", torch.equal(positives, torch.arange(6, 12)))
            check("triplet_mining random negatives in positive half", bool(((negatives >= 6) & (negatives < 12)).all()))
            check("triplet_mining random avoids true positive index", not torch.any(negatives == positives).item())
        else:
            skip_checks(5, "triplet mining returned None")
    except Exception as exc:
        skip_checks(6, f"random triplet_mining raised {exc!r}")

    try:
        args = types.SimpleNamespace(triplet_mine=True, walk=2)
        model = types.SimpleNamespace(hparams={"args": args}, level_of_distance=-1)
        z_a = F.normalize(torch.randn(6, 64), dim=1)
        z_b = F.normalize(torch.randn(6, 64), dim=1)
        batch = {
            "pos": torch.tensor([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1], [20.0, 20.0], [20.1, 20.1]]),
            "ptr": torch.tensor([0, 2, 4, 6]),
        }
        embeddings, anchors, positives, negatives = FullModel.triplet_mining(model, batch, z_a, z_b)
        check("triplet_mining spatial output not None", negatives is not None)
        if negatives is not None:
            check("triplet_mining spatial negatives shape", tuple(negatives.shape) == (6,), f"got {tuple(negatives.shape)}")
            check("triplet_mining spatial repeats each graph negative by walk", torch.equal(negatives[0::2], negatives[1::2]))
            check("triplet_mining spatial negatives are shifted to positive half", bool((negatives >= 6).all()))
            check("triplet_mining spatial embeddings remain concatenated", tuple(embeddings.shape) == (12, 64))
        else:
            skip_checks(4, "spatial triplet mining returned None")
    except Exception as exc:
        skip_checks(5, f"spatial triplet_mining raised {exc!r}")

    try:
        class Batch(dict):
            def to_data_list(self):
                return ["walk"]

        class IdentityAugmentor:
            def __call__(self, x):
                return x

        class FakeLoss:
            def __call__(self, embeddings, indices_tuple):
                anchors, positives, negatives = indices_tuple
                return (embeddings[anchors] - embeddings[positives]).pow(2).mean() + negatives.float().mean() * 0.0

        class WalkHarness:
            def __init__(self):
                self.hparams = {"args": types.SimpleNamespace(triplet_mine=False, walk=2)}
                self.augmentor = IdentityAugmentor()
                self.val_process = IdentityAugmentor()
                self.loss_function = FakeLoss()
                self.batch_size = 2
                self.level_of_distance = -1
                self.train_a, self.train_b, self.val_a, self.val_b = [], [], [], []
                self.test_a, self.test_b = [], []
                self.gt_ori_train, self.gt_ori_val, self.gt_ori_test = [], [], []
                self.train_loss, self.val_loss, self.test_loss = [], [], []
                self.logged = []

            def forward(self, x, edge_index, img='sat'):
                base = torch.arange(x.shape[0] * 64, dtype=torch.float32).view(x.shape[0], 64)
                if img == 'pov':
                    base = base + 1.0
                return F.normalize(base + 0.01, dim=1)

            def triplet_mining(self, batch, z_a, z_b):
                return FullModel.triplet_mining(self, batch, z_a, z_b)

            def log(self, *args, **kwargs):
                self.logged.append((args, kwargs))

        harness = WalkHarness()
        batch = Batch({
            "sat_image": torch.randn(4, 3, 8, 8),
            "pov_image": torch.randn(4, 3, 8, 8),
            "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
            "yaws_image": torch.tensor([0.0, 10.0, 20.0, 30.0]),
            "ptr": torch.tensor([0, 2, 4]),
            "pos": torch.randn(4, 2),
        })
        loss = FullModel.walk_step(harness, batch, batch_idx=0, stage='train')
        check("walk_step output not None", loss is not None)
        if loss is not None:
            check("walk_step loss scalar", loss.ndim == 0)
            check("walk_step loss finite", torch.isfinite(loss).item())
            check("walk_step logs stage loss", len(harness.logged) == 1 and harness.logged[0][0][0] == "train_loss")
            check("walk_step stores pointer-selected train embeddings", len(harness.train_a) == 1 and harness.train_a[0].shape == (2, 64))
            check("walk_step stores train yaw pointers", len(harness.gt_ori_train) == 1 and harness.gt_ori_train[0].shape == (2,))
        else:
            skip_checks(5, "walk_step returned None")
    except Exception as exc:
        skip_checks(6, f"walk_step raised {exc!r}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
