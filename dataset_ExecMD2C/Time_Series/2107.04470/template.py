import torch
from torch import nn


## Convolutional blocks
class base_model_blk1(nn.Module):
    def __init__(self, configs):
        super(base_model_blk1, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        return x


class base_model_blk2(nn.Module):
    def __init__(self, configs):
        super(base_model_blk2, self).__init__()

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x_in):
        x = self.conv_block2(x_in)
        return x


class base_model_blk3(nn.Module):
    def __init__(self, configs):
        super(base_model_blk3, self).__init__()
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x_in):
        x = self.conv_block3(x_in)
        return x



class cnn_feature_extractor(nn.Module):
    def __init__(self, configs):
        super(cnn_feature_extractor, self).__init__()
        self.conv_block1_shared = base_model_blk1(configs)
        self.conv_block2_shared = base_model_blk2(configs)
        self.conv_block3_shared = base_model_blk3(configs)

    def forward(self, input):
        out = self.conv_block1_shared(input)
        out = self.conv_block2_shared(out)
        out = self.conv_block3_shared(out)
        
        return out

class Classifier(nn.Module):
    def __init__(self, configs):
        super(Classifier, self).__init__()
        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, input):
        logits = self.logits(input)
        return logits


class Discriminator(nn.Module):
    def __init__(self, configs):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 1)
        )

    def forward(self, input):
        out = self.layer(input)
        return out



class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
        [TODO] Build a temporal self-attention representation for CNN feature maps.

        Input:
            x: (batch, channels, temporal_width) - convolutional EEG features.

        Output: (batch, channels * temporal_width) - flattened attention-weighted features.

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

    print("=" * 70)
    print("ADAST: Automated Test Suite - 1 ablated target")
    print("=" * 70)

    try:
        module = Self_Attn(6)
        x = torch.randn(2, 6, 7, requires_grad=True)
        output = module(x)
        check("Self_Attn output not None", output is not None)
        if output is not None:
            check("Self_Attn output shape", tuple(output.shape) == (2, 42),
                  f"expected (2, 42), got {tuple(output.shape)}")
            check("Self_Attn output finite", torch.isfinite(output).all().item())
            output.sum().backward()
            check("Self_Attn attention gradients", module.query_conv.weight.grad is not None and
                  module.query_conv.weight.grad.abs().sum().item() > 0 and x.grad is not None)
        else:
            skip_checks(3, "Self_Attn.forward returned None")
    except Exception as exc:
        print(f"  [Self_Attn.forward] ERROR - {exc}")
        skip_checks(4, "Self_Attn.forward raised an exception")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)

