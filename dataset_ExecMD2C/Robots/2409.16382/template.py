# ============================================================
# ground_truth.py - LetsPlay4Emotion Core Model Components
# Source: LetsPlay4Emotion-main
#
# Contains ONLY the model wrapper and custom SlowFast pathway
# component used by the video emotion/pain classifier.
# No training loop, dataset loading, mesh generation, rendering,
# checkpoint download, or CLI code is included.
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Precision, Recall, F1Score, Accuracy, ConfusionMatrix, AUROC


# --- [Original file: model/model_slowfast.py] ---
class NeuralNetworkModel(LightningModule):
    def __init__(self, num_classes, model_type, train_dataset_file, val_dataset_file, nn_model, augmentation_train,
                 augmentation_val, clip_duration, batch_size, video_path_prefix):
        super().__init__()
        self.video_path_prefix = video_path_prefix
        self.model_type = model_type
        self.train_dataset_file = train_dataset_file
        self.val_dataset_file = val_dataset_file
        self.augmentation_train = augmentation_train
        self.augmentation_val = augmentation_val
        self.conv_layers = nn_model
        self.lr = 0.01
        self.batch_size = batch_size
        self.num_worker = 8
        self.clip_duration = clip_duration
        self.num_classes = num_classes

        # Define metrics
        self.f1_score = F1Score(task=self.model_type, num_classes=self.num_classes)
        self.accuracy = Accuracy(task=self.model_type, num_classes=self.num_classes)
        self.auroc = AUROC(task=self.model_type, num_classes=self.num_classes)
        self.precision = Precision(task=self.model_type, average='macro', num_classes=self.num_classes)
        self.recall = Recall(task=self.model_type, average='macro', num_classes=self.num_classes)
        self.confusion_matrix = ConfusionMatrix(task=self.model_type, num_classes=self.num_classes, normalize="true")

        # Define loss function based on model type
        if self.model_type == "binary":
            self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([4.004]))
        else:
            self.loss = nn.CrossEntropyLoss()

        self.validation_output_list = []

    def forward(self, x):
        return self.conv_layers(x)


# --- [Original file: model/model_slowfast.py] ---
class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        """
        [TODO] Convert a video tensor into the two pathway inputs expected by a SlowFast model.

        Input:
            frames: (batch, time, channels, height, width) - temporally ordered video frames.

        Output:
            list with two tensors:
                slow_pathway: (batch, time / 4, channels, height, width) - frames sampled
                    uniformly along the temporal dimension using the SlowFast alpha ratio.
                fast_pathway: (batch, time, channels, height, width) - the complete original
                    frame tensor.

"""
        pass


# ============================================================
# __main__: Automated test suite for 1 ablated function
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

    print("=" * 70)
    print("LetsPlay4Emotion SlowFast Pathway Packing")
    print("Automated Test Suite - 1 ablated function")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/1: PackPathway.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/1] PackPathway.forward - SlowFast temporal pathway split")
    try:
        module = PackPathway().to(device)
        frames = torch.arange(2 * 16 * 3 * 4 * 5, dtype=torch.float32, device=device).reshape(2, 16, 3, 4, 5)
        output = module(frames)
        check("PackPathway output not None", output is not None)
        if output is not None:
            check("PackPathway returns two pathways", isinstance(output, list) and len(output) == 2)
            if isinstance(output, list) and len(output) == 2:
                slow_pathway, fast_pathway = output
                check("PackPathway slow shape", tuple(slow_pathway.shape) == (2, 4, 3, 4, 5),
                      f"expected (2, 4, 3, 4, 5), got {tuple(slow_pathway.shape)}")
                check("PackPathway fast shape", tuple(fast_pathway.shape) == (2, 16, 3, 4, 5),
                      f"expected (2, 16, 3, 4, 5), got {tuple(fast_pathway.shape)}")
                check("PackPathway fast aliases full input", torch.equal(fast_pathway, frames))
                expected_indices = torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // 4).long()
                expected_slow = torch.index_select(frames, 1, expected_indices)
                check("PackPathway slow samples alpha-4 indices", torch.equal(slow_pathway, expected_slow))
                check("PackPathway outputs finite",
                      torch.isfinite(slow_pathway).all().item() and torch.isfinite(fast_pathway).all().item())
            else:
                skip_checks(5, "PackPathway.forward did not return a two-item list")
        else:
            skip_checks(6, "PackPathway.forward returned None")
    except Exception as exc:
        print(f"  [PackPathway.forward] ERROR: {exc}")
        skip_checks(7, "PackPathway.forward raised an exception")
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
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
