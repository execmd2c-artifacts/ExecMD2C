import collections
import dataclasses
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import functional as F


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


class _NewEmptyTensorOp(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, new_shape):
    ctx.shape = x.shape
    return x.new_empty(new_shape)

  @staticmethod
  def backward(ctx, grad):
    shape = ctx.shape
    return _NewEmptyTensorOp.apply(grad, shape), None


@dataclasses.dataclass
class Node:
  inp: any
  module: nn.Module
  activated: bool
  stride: int
  dim: int

  def __hash__(self):
    return hash(self.module)


class NetFactory(nn.Module):

  def __init__(self):
    super().__init__()
    self.nodes = []
    self.skips = {}
    self.tags = {}

  def tag(self, node, name):
    self.tags[node] = name

  def input(self, in_dim=3, stride=1, activated=True):
    assert not self.nodes
    n = Node(inp=None, module=None, activated=activated, stride=stride, dim=in_dim)
    self.nodes.append(n)
    return n

  def _add(self, node):
    self.nodes.append(node)
    return node

  def _activate(self, node):
    if node.activated:
      return node
    return self._add(
        dataclasses.replace(
            node,
            inp=node,
            module=nn.Sequential(nn.BatchNorm2d(node.dim), nn.LeakyReLU()),
            activated=True
        )
    )

  def _conv(self, node, out_dim=None, stride=1, rate=1, kernel=3):
    node = self._activate(node)
    if out_dim is None:
      out_dim = node.dim
    padding = (kernel - 1) // 2 * rate
    return self._add(
        dataclasses.replace(
            node,
            inp=node,
            module=nn.Conv2d(
                node.dim, out_dim, kernel, stride=stride, dilation=rate, padding=padding
            ),
            activated=False,
            dim=out_dim,
            stride=node.stride * stride
        )
    )

  def _interp(self, node):
    return self._add(
        dataclasses.replace(
            node,
            inp=node,
            module=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            stride=node.stride // 2
        )
    )

  def _lateral(self, node, out_dim=None):
    if out_dim is None:
      out_dim = node.dim
    if out_dim == node.dim:
      return node
    return self._conv(node, out_dim=out_dim, kernel=1)

  def output(self, node, out_dim):
    return self._conv(node, out_dim=out_dim, kernel=1)

  def downscale(self, node, out_dim):
    return self._conv(node, out_dim, stride=2)

  def upsample(self, node, skip, out_dim):
    skip = self._lateral(skip, out_dim=out_dim)
    node = self._lateral(node, out_dim=out_dim)
    node = self._interp(node)
    self.skips[node] = skip
    return node

  def layer(self, node, out_dim=None, rate=1):
    if out_dim is None:
      out_dim = node.dim
    skip = self._lateral(node, out_dim=out_dim)
    node = self._conv(node, rate=rate)
    node = self._conv(node, rate=rate)
    self.skips[node] = skip
    return node

  def block(self, node, rates):
    for r in [int(r) for r in rates]:
      node = self.layer(node, rate=r)
    return node

  def bake(self):
    self.modules = nn.ModuleList(n.module for n in self.nodes if n.module is not None)
    return self

  def forward(self, x):
    """
    [TODO] Execute the baked graph of convolution, residual, upsample, and tag nodes.

    Input:
        x: (batch, channels, height, width) - input tensor for the first graph node.

    Output:
        Tensor or dict[str, Tensor] - final tensor when no tags exist, otherwise all tagged features.

"""
    pass


def hdrn_alpha_base(num_channels):
  net = NetFactory()
  x = net.input()
  x = net.downscale(x, num_channels)
  x = net.downscale(x, num_channels)
  x4 = x = net.block(x, '111')
  x = net.downscale(x, num_channels * 2)
  x8 = x = net.block(x, '1111')
  x = net.downscale(x, num_channels * 4)
  x = net.block(x, '12591259')
  x = net.upsample(x, x8, num_channels // 2)
  x = net.upsample(x, x4, num_channels // 2)
  return net.bake()


def make_process_cost_volume(num_disparities):
  net = NetFactory()
  x = net.input(in_dim=num_disparities, stride=4, activated=True)
  x = net.block(x, '1259')
  x = net.output(x, out_dim=num_disparities)
  return net.bake()


class HdrnAlphaStereo(nn.Module):

  def __init__(self, hparams):
    super().__init__()

    self.num_disparities = hparams.max_disparity
    self.internal_scale = hparams.cost_volume_downsample_factor
    self.internal_num_disparities = self.num_disparities // self.internal_scale
    assert self.internal_scale in [4, 8, 16]

    self.feature_extractor = hdrn_alpha_base(hparams.fe_internal_features)
    self.cost_volume = DotProductCostVolume(self.internal_num_disparities)
    self.process_cost_volume = make_process_cost_volume(self.internal_num_disparities)

    self.soft_argmin = SoftArgmin()

  def forward(self, left_image, right_image):
    """
    [TODO] Estimate low-resolution stereo disparity from left and right RGB images.

    Input:
        left_image: (batch, 3, height, width) - left image.
        right_image: (batch, 3, height, width) - right image.

    Output:
        (batch, 1, height / 4, width / 4) - continuous small-disparity prediction.

"""
    pass


class StereoBackbone(nn.Module):

  def __init__(self, hparams, in_channels=3):
    super().__init__()

    def make_rgb_stem():
      net = NetFactory()
      x = net.input(in_dim=3, stride=1, activated=True)
      x = net.downscale(x, 32)
      x = net.downscale(x, 32)
      return net.bake()

    def make_disp_features():
      net = NetFactory()
      x = net.input(in_dim=1, stride=1, activated=False)
      x = net.layer(x, 32, rate=5)
      return net.bake()

    self.rgb_stem = make_rgb_stem()
    self.stereo_stem = HdrnAlphaStereo(hparams)
    self.disp_features = make_disp_features()

    def make_rgbd_backbone(num_channels=64, out_dim=64):
      net = NetFactory()
      x = net.input(in_dim=64, activated=True, stride=4)
      x = net._lateral(x, out_dim=num_channels)
      x4 = x = net.block(x, '111')
      x = net.downscale(x, num_channels * 2)
      x8 = x = net.block(x, '1111')
      x = net.downscale(x, num_channels * 4)
      x = net.block(x, '12591259')
      net.tag(net.output(x, out_dim), 'p4')
      x = net.upsample(x, x8, out_dim)
      net.tag(x, 'p3')
      x = net.upsample(x, x4, out_dim)
      net.tag(x, 'p2')
      return net.bake()

    self.rgbd_backbone = make_rgbd_backbone()

  def forward(self, stacked_img, step, robot_joint_angles=None):
    """
    [TODO] Fuse stereo disparity features with left-image RGB features for SimNet.

    Input:
        stacked_img: (batch, 6, height, width) - concatenated left and right RGB images.
        step: scalar/int - training step argument kept for interface compatibility.
        robot_joint_angles: optional tensor - unused in this backbone.

    Output:
        dict with:
            p2: (batch, 64, height / 4, width / 4)
            p3: (batch, 64, height / 8, width / 8)
            p4: (batch, 64, height / 16, width / 16)
            small_disp: (batch, 1, height / 4, width / 4)

"""
    pass

  @property
  def out_channels(self):
    return 32

  @property
  def stride(self):
    return 4  # = stride 2 conv -> stride 2 max pool


@torch.jit.script
def cost_volume(left, right, num_disparities: int, is_right: bool):
  """
  [TODO] Build a channel-preserving stereo cost volume by shifting feature maps.

  Input:
      left: (batch, channels, height, width) - left feature map.
      right: (batch, channels, height, width) - right feature map.
      num_disparities: number of disparity shifts.
      is_right: whether to use the right-view indexing convention.

  Output:
      (batch, channels, num_disparities, height, width) - shifted matching costs.

"""
  pass


class CostVolume(nn.Module):
  """Compute cost volume using cross correlation of left and right feature maps"""

  def __init__(self, num_disparities, is_right=False):
    super().__init__()
    self.num_disparities = num_disparities
    self.is_right = is_right

  def forward(self, left, right):
    """
    [TODO] Dispatch cross-correlation cost-volume computation.

    Input:
        left: (batch, channels, height, width) - left features.
        right: (batch, channels, height, width) - right features.

    Output:
        (batch, channels, num_disparities, height, width) - cost volume.

"""
    pass

  @torch.jit.unused
  def forward_with_amp(self, left, right):
    """This operation is unstable at float16, so compute at float32 even when using mixed precision"""
    with torch.cuda.amp.autocast(enabled=False):
      left = left.to(torch.float32)
      right = right.to(torch.float32)
      output = cost_volume(left, right, self.num_disparities, self.is_right)
      output = torch.clamp(output, -1e3, 1e3)
      return output


@torch.jit.script
def dot_product_cost_volume(left, right, num_disparities: int, is_right: bool):
  """
  [TODO] Build a dot-product stereo cost volume over disparity shifts.

  Input:
      left: (batch, channels, height, width) - left feature map.
      right: (batch, channels, height, width) - right feature map.
      num_disparities: number of disparity shifts.
      is_right: whether to use the right-view indexing convention.

  Output:
      (batch, num_disparities, height, width) - per-disparity average channel similarity.

"""
  pass


class DotProductCostVolume(nn.Module):
  """Compute cost volume using dot product of left and right feature maps"""

  def __init__(self, num_disparities, is_right=False):
    super().__init__()
    self.num_disparities = num_disparities
    self.is_right = is_right

  def forward(self, left, right):
    """
    [TODO] Compute the configured dot-product cost volume.

    Input:
        left: (batch, channels, height, width) - left features.
        right: (batch, channels, height, width) - right features.

    Output:
        (batch, num_disparities, height, width) - dot-product cost volume.

"""
    pass

  @torch.jit.unused
  def forward_with_amp(self, left, right):
    """This operation is unstable at float16, so compute at float32 even when using mixed precision"""
    with torch.cuda.amp.autocast(enabled=False):
      left = left.to(torch.float32)
      right = right.to(torch.float32)
      output = dot_product_cost_volume(left, right, self.num_disparities, self.is_right)
      output = torch.clamp(output, -1e3, 1e3)
      return output


@torch.jit.script
def soft_argmin(input):
  """
  [TODO] Convert a disparity cost volume into a continuous disparity estimate.

  Input:
      input: (batch, disparities, height, width) - lower cost means better match.

  Output:
      (batch, 1, height, width) - expected disparity index.

"""
  pass


class SoftArgmin(nn.Module):
  """Compute soft argmin operation for given cost volume"""

  def forward(self, input):
    """
    [TODO] Apply differentiable soft argmin to a cost volume.

    Input:
        input: (batch, disparities, height, width) - stereo matching costs.

    Output:
        (batch, 1, height, width) - continuous disparity.

"""
    pass


@torch.jit.script
def matchability(input):
  softmin = F.softmin(input, dim=1)
  log_softmin = F.log_softmax(-input, dim=1)
  output = torch.sum(softmin * log_softmin, dim=1, keepdim=True)
  return output


class Matchability(nn.Module):
  """Compute disparity matchability value from https://arxiv.org/abs/2008.04800"""

  def forward(self, input):
    if torch.jit.is_scripting():
      # Torchscript generation can't handle mixed precision, so always compute at float32.
      return matchability(input)
    else:
      return self.forward_with_amp(input)

  @torch.jit.unused
  def forward_with_amp(self, input):
    """This operation is unstable at float16, so compute at float32 even when using mixed precision"""
    with torch.cuda.amp.autocast(enabled=False):
      input = input.to(torch.float32)
      return matchability(input)


class MaskedL1Loss(nn.Module):

  def __init__(self, centroid_threshold=0.3, downscale_factor=8):
    super().__init__()
    self.loss = nn.L1Loss(reduction='none')
    self.centroid_threshold = centroid_threshold
    self.downscale_factor = downscale_factor

  def forward(self, output, target, valid_mask):
    '''
        output: [N,16,H,W]
        target: [N,16,H,W]
        valid_mask: [N,H,W]
        '''
    valid_count = torch.sum(
        valid_mask[:, ::self.downscale_factor, ::self.downscale_factor] > self.centroid_threshold
    )
    loss = self.loss(output, target)
    if len(output.shape) == 4:
      loss = torch.sum(loss, dim=1)
    loss[valid_mask[:, ::self.downscale_factor, ::self.downscale_factor] < self.centroid_threshold
        ] = 0.0
    if valid_count == 0:
      return torch.sum(loss)
    return torch.sum(loss) / valid_count


class MSELoss(nn.Module):

  def __init__(self):
    super().__init__()
    self.loss = nn.MSELoss(reduction='none')

  def forward(self, output, target):
    '''
        output: [N,H,W]
        target: [N,H,W]
        ignore_mask: [N,H,W]
        '''
    loss = self.loss(output, target)
    return torch.mean(loss)


class MaskedMSELoss(nn.Module):

  def __init__(self):
    super().__init__()
    self.loss = nn.MSELoss(reduction='none')

  def forward(self, output, target, ignore_mask):
    '''
        output: [N,H,W]
        target: [N,H,W]
        ignore_mask: [N,H,W]
        '''
    valid_sum = torch.sum(torch.logical_not(ignore_mask))
    loss = self.loss(output, target)
    loss[ignore_mask > 0] = 0.0
    return torch.sum(loss) / valid_sum


class DepthOutput:

  def __init__(self, depth_pred, loss_multiplier):
    self.depth_pred = depth_pred
    self.is_numpy = False
    self.loss = nn.SmoothL1Loss()
    self.loss_multiplier = loss_multiplier


class SegmentationOutput:

  def __init__(self, seg_pred, hparams):
    self.seg_pred = seg_pred
    self.is_numpy = False
    self.hparams = hparams


class PoseOutput:

  def __init__(self, heatmap, vertex_field, z_centroid_field, hparams):
    self.heatmap = heatmap
    self.vertex_field = vertex_field
    self.z_centroid_field = z_centroid_field
    self.is_numpy = False
    self.hparams = hparams


class OBBOutput:

  def __init__(self, heatmap, vertex_field, z_centroid_field, cov_field, hparams):
    self.heatmap = heatmap
    self.vertex_field = vertex_field
    self.z_centroid_field = z_centroid_field
    self.cov_field = cov_field
    self.is_numpy = False
    self.hparams = hparams


depth_outputs = types.SimpleNamespace(DepthOutput=DepthOutput)
segmentation_outputs = types.SimpleNamespace(SegmentationOutput=SegmentationOutput)
pose_outputs = types.SimpleNamespace(PoseOutput=PoseOutput)
obb_outputs = types.SimpleNamespace(OBBOutput=OBBOutput)
simplenet = types.SimpleNamespace(StereoBackbone=StereoBackbone)


MODEL_SEM_SEG_HEAD_IN_FEATURES = ['p2', 'p3', 'p4']
MODEL_POSE_HEAD_IN_FEATURES = ['p3', 'p4']

MODEL_SEM_SEG_HEAD_IGNORE_VALUE = 255
MODEL_SEM_SEG_HEAD_COMMON_STRIDE = 4
MODEL_POSE_HEAD_COMMON_STRIDE = 8
MODEL_SEM_SEG_HEAD_LOSS_WEIGHT = 1.0


def c2_msra_fill(module: nn.Module) -> None:
  """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
  nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
  if module.bias is not None:
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
    #  torch.Tensor]`.
    nn.init.constant_(module.bias, 0)


class Conv2d(torch.nn.Conv2d):
  """
  A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
  """

  def __init__(self, *args, **kwargs):
    """
    Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

    Args:
      norm (nn.Module, optional): a normalization layer
      activation (callable(Tensor) -> Tensor): a callable activation function

    It assumes that norm layer is used before activation.
    """
    norm = kwargs.pop("norm", None)
    activation = kwargs.pop("activation", None)
    super().__init__(*args, **kwargs)

    self.norm = norm
    self.activation = activation

  def forward(self, x):
    if x.numel() == 0 and self.training:
      # https://github.com/pytorch/pytorch/issues/12013
      assert not isinstance(
          self.norm, torch.nn.SyncBatchNorm
      ), "SyncBatchNorm does not support empty inputs!"

    if x.numel() == 0 and TORCH_VERSION <= (1, 4):
      assert not isinstance(
          self.norm, torch.nn.GroupNorm
      ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
      # When input is empty, we want to return a empty tensor with "correct" shape,
      # So that the following operations will not panic
      # if they check for the shape of the tensor.
      # This computes the height and width of the output tensor
      output_shape = [(i + 2 * p - (di * (k - 1) + 1)) // s + 1 for i, p, di, k, s in
                      zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
      output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
      empty = _NewEmptyTensorOp.apply(x, output_shape)
      if self.training:
        # This is to make DDP happy.
        # DDP expects all workers to have gradient w.r.t the same set of parameters.
        _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
        return empty + _dummy
      else:
        return empty

    x = super().forward(x)
    if self.norm is not None:
      x = self.norm(x)
    if self.activation is not None:
      x = self.activation(x)
    return x


def get_norm(norm, out_channels):
  """
  Args:
    norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
      or a callable that takes a channel number and returns
      the normalization layer as a nn.Module.

  Returns:
    nn.Module or None: the normalization layer
  """
  if out_channels == 32:
    N = 16
  else:
    N = 32
  if isinstance(norm, str):
    if len(norm) == 0:
      return None
    norm = {
        "BN": torch.nn.BatchNorm2d,
        #"SyncBN": NaiveSyncBatchNorm,
        #"FrozenBN": FrozenBatchNorm2d,
        "GN": lambda channels: nn.GroupNorm(N, channels),
        #"nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
    }[norm]
  return norm(out_channels)


class SemSegFPNHead(nn.Module):
  """
  A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
  (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
  all levels of the FPN into single output.
  """

  def __init__(self, input_shape, num_classes, model_norm='BN', num_filters_scale=4):
    super().__init__()
    MODEL_SEM_SEG_HEAD_NORM = model_norm
    MODEL_SEM_SEG_HEAD_CONVS_DIM = 128 // num_filters_scale

    self.in_features = MODEL_SEM_SEG_HEAD_IN_FEATURES
    feature_strides = {k: v.stride for k, v in input_shape.items()}
    feature_channels = {k: v.channels for k, v in input_shape.items()}
    self_ignore_value = MODEL_SEM_SEG_HEAD_IGNORE_VALUE
    conv_dims = MODEL_SEM_SEG_HEAD_CONVS_DIM
    self.common_stride = MODEL_SEM_SEG_HEAD_COMMON_STRIDE
    norm = MODEL_SEM_SEG_HEAD_NORM
    self.bilinear_upsample = nn.Upsample(
        scale_factor=self.common_stride, mode="bilinear", align_corners=False
    )

    self.scale_heads = []
    for in_feature in self.in_features:
      head_ops = []
      head_length = max(1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)))
      for k in range(head_length):
        norm_module = get_norm(norm, conv_dims)
        conv = Conv2d(
            feature_channels[in_feature] if k == 0 else conv_dims,
            conv_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm,
            norm=norm_module,
            activation=F.relu,
        )
        c2_msra_fill(conv)
        head_ops.append(conv)
        if feature_strides[in_feature] != self.common_stride:
          head_ops.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
      self.scale_heads.append(nn.Sequential(*head_ops))
      self.add_module(in_feature, self.scale_heads[-1])
    self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
    c2_msra_fill(self.predictor)

  def forward(self, features, targets=None):
    """
    [TODO] Produce full-resolution semantic/depth logits from FPN features.

    Input:
        features: dict containing p2, p3, p4 tensors with strides 4, 8, and 16.
        targets: optional target argument retained for interface compatibility.

    Output:
        (batch, num_classes, input_height, input_width) - upsampled prediction map.

"""
    pass

  def layers(self, features):
    """
    [TODO] Merge p2/p3/p4 feature maps into a common-stride prediction map.

    Input:
        features: dict[str, Tensor] - selected FPN features.

    Output:
        (batch, num_classes, height / common_stride, width / common_stride)

"""
    pass

  def losses(self, predictions, targets):
    predictions = F.interpolate(
        predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
    )
    loss = F.cross_entropy(predictions, targets, reduction="mean", ignore_index=self.ignore_value)
    return loss


class PoseFPNHead(nn.Module):
  """
  A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
  (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
  all levels of the FPN into single output.
  """

  def __init__(self, input_shape, num_classes, model_norm='BN', num_filters_scale=4):
    super().__init__()
    MODEL_SEM_SEG_HEAD_NORM = model_norm
    MODEL_SEM_SEG_HEAD_CONVS_DIM = 128 // num_filters_scale
    self.in_features = MODEL_POSE_HEAD_IN_FEATURES
    feature_strides = {k: v.stride for k, v in input_shape.items()}
    feature_channels = {k: v.channels for k, v in input_shape.items()}
    self_ignore_value = MODEL_SEM_SEG_HEAD_IGNORE_VALUE
    conv_dims = MODEL_SEM_SEG_HEAD_CONVS_DIM
    self.common_stride = MODEL_POSE_HEAD_COMMON_STRIDE
    norm = MODEL_SEM_SEG_HEAD_NORM

    self.scale_heads = []
    for in_feature in self.in_features:
      head_ops = []
      head_length = max(1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)))
      for k in range(head_length):
        norm_module = get_norm(norm, conv_dims)
        conv = Conv2d(
            feature_channels[in_feature] if k == 0 else conv_dims,
            conv_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm,
            norm=norm_module,
            activation=F.relu,
        )
        c2_msra_fill(conv)
        head_ops.append(conv)
        if feature_strides[in_feature] != self.common_stride:
          head_ops.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
      self.scale_heads.append(nn.Sequential(*head_ops))
      self.add_module(in_feature, self.scale_heads[-1])
    self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
    c2_msra_fill(self.predictor)

  def forward(self, features, targets=None):
    """
    [TODO] Produce pose/OBB head logits from stride-8 FPN features.

    Input:
        features: dict containing p3 and p4 tensors with strides 8 and 16.
        targets: optional target argument retained for interface compatibility.

    Output:
        (batch, num_classes, input_height / 8, input_width / 8) - pose prediction map.

"""
    pass

  def layers(self, features):
    """
    [TODO] Merge p3/p4 features for pose-style prediction heads.

    Input:
        features: dict[str, Tensor] - selected stride-8 and stride-16 FPN features.

    Output:
        (batch, num_classes, input_height / common_stride, input_width / common_stride)

"""
    pass


class ShapeSpec(collections.namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
  """
  A simple structure that contains basic shape specification about a tensor.
  It is often used as the auxiliary inputs/outputs of models,
  to obtain the shape inference ability among pytorch modules.

  Attributes:
    channels:
    height:
    width:
    stride:
  """

  def __new__(cls, *, channels=None, height=None, width=None, stride=None):
    return super().__new__(cls, channels, height, width, stride)


def res_fpn(hparams):
  return PanopticNet(hparams)


class DepthHead(nn.Module):

  def __init__(self, backbone_output_shape_4x, backbone_output_shape_8x, hparams):
    super().__init__()
    self.head = SemSegFPNHead(
        backbone_output_shape_4x,
        num_classes=1,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    self.hparams = hparams

  def forward(self, features):
    depth_pred = self.head.forward(features)
    depth_pred = depth_pred.squeeze(dim=1)
    return depth_outputs.DepthOutput(depth_pred, self.hparams.loss_depth_refine_mult)


class SegmentationHead(nn.Module):

  def __init__(self, backbone_output_shape_4x, backbone_output_shape_8x, num_classes, hparams):
    super().__init__()
    self.head = SemSegFPNHead(
        backbone_output_shape_4x,
        num_classes=num_classes,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    self.hparams = hparams

  def forward(self, features):
    pred = self.head.forward(features)
    return segmentation_outputs.SegmentationOutput(pred, self.hparams)


class PoseHead(nn.Module):

  def __init__(self, backbone_output_shape_4x, backbone_output_shape_8x, hparams):
    super().__init__()
    self.hparams = hparams
    self.heatmap_head = SemSegFPNHead(
        backbone_output_shape_4x,
        num_classes=1,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    self.vertex_head = PoseFPNHead(
        backbone_output_shape_8x,
        num_classes=16,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )
    self.z_centroid_head = PoseFPNHead(
        backbone_output_shape_8x,
        num_classes=1,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )

  def forward(self, features):
    z_centroid_output = self.z_centroid_head.forward(features).squeeze(dim=1)
    heatmap_output = self.heatmap_head.forward(features).squeeze(dim=1)
    vertex_output = self.vertex_head.forward(features)
    return pose_outputs.PoseOutput(heatmap_output, vertex_output, z_centroid_output, self.hparams)


class OBBHead(nn.Module):

  def __init__(self, backbone_output_shape_4x, backbone_output_shape_8x, hparams):
    super().__init__()
    self.hparams = hparams
    self.heatmap_head = SemSegFPNHead(
        backbone_output_shape_4x,
        num_classes=1,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )

    self.vertex_head = PoseFPNHead(
        backbone_output_shape_8x,
        num_classes=16,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )

    self.z_centroid_head = PoseFPNHead(
        backbone_output_shape_8x,
        num_classes=1,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )

    self.rotation_head = PoseFPNHead(
        backbone_output_shape_8x,
        num_classes=6,
        model_norm=hparams.model_norm,
        num_filters_scale=hparams.num_filters_scale
    )

  def forward(self, features):
    z_centroid_output = self.z_centroid_head.forward(features).squeeze(dim=1)
    heatmap_output = self.heatmap_head.forward(features).squeeze(dim=1)
    vertex_output = self.vertex_head.forward(features)
    rotation_output = self.rotation_head.forward(features)
    return obb_outputs.OBBOutput(
        heatmap_output, vertex_output, z_centroid_output, rotation_output, self.hparams
    )


class PanopticNet(nn.Module):

  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    self.backbone = simplenet.StereoBackbone(hparams)
    # ResFPN used p2,p3,p4,p5 (64 channels)
    # DRN uses only p2,p3,p4 (no need for p5 since dilation increases striding naturally)
    backbone_output_shape_4x = {
        #'p0': ShapeSpec(channels=64, height=None, width=None, stride=1),
        #'p1': ShapeSpec(channels=64, height=None, width=None, stride=2),
        'p2': ShapeSpec(channels=64, height=None, width=None, stride=4),
        'p3': ShapeSpec(channels=64, height=None, width=None, stride=8),
        'p4': ShapeSpec(channels=64, height=None, width=None, stride=16),
        #'p5': ShapeSpec(channels=64, height=None, width=None, stride=32),
    }

    backbone_output_shape_8x = {
        #'p0': ShapeSpec(channels=64, height=None, width=None, stride=1),
        #'p1': ShapeSpec(channels=64, height=None, width=None, stride=2),
        #'p2': ShapeSpec(channels=64, height=None, width=None, stride=4),
        'p3': ShapeSpec(channels=64, height=None, width=None, stride=8),
        'p4': ShapeSpec(channels=64, height=None, width=None, stride=16),
        #'p5': ShapeSpec(channels=64, height=None, width=None, stride=32),
    }

    # Add depth head.
    self.depth_head = DepthHead(backbone_output_shape_4x, backbone_output_shape_8x, hparams)
    # Add segmentation head.
    self.seg_head = SegmentationHead(backbone_output_shape_4x, backbone_output_shape_8x, 5, hparams)
    # Add pose heads.
    self.pose_head = OBBHead(backbone_output_shape_4x, backbone_output_shape_8x, hparams)

  def forward(self, image, step):
    """
    [TODO] Run the full SimNet multi-task forward pass.

    Input:
        image: (batch, 6, height, width) - stacked left/right RGB stereo image.
        step: scalar/int - training step argument passed to the backbone.

    Output:
        tuple of six outputs:
            segmentation output with (batch, 5, height, width)
            refined depth output with (batch, height, width)
            small disparity output with (batch, height / 4, width / 4)
            OBB output with heatmap, vertex, z-centroid, and covariance fields
            box output placeholder
            keypoint output placeholder

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
  print("SimNet: stereo panoptic model code-reproduction benchmark")
  print("=" * 70)

  class HParams:
    max_disparity = 16
    cost_volume_downsample_factor = 4
    fe_internal_features = 4
    model_norm = "GN"
    num_filters_scale = 4
    loss_depth_mult = 1.0
    loss_depth_refine_mult = 1.0
    loss_seg_mult = 1.0
    loss_vertex_mult = 0.1
    loss_rotation_mult = 0.1
    loss_heatmap_mult = 100.0
    loss_z_centroid_mult = 0.1
    frozen_stereo_checkpoint = None

  try:
    left = torch.tensor([[[[1., 2., 3., 4.],
                           [5., 6., 7., 8.]]]])
    right = torch.tensor([[[[2., 3., 4., 5.],
                            [6., 7., 8., 9.]]]])
    cv = cost_volume(left, right, 3, False)
    check("cost_volume output not None", cv is not None)
    if cv is not None:
      check("cost_volume shape", tuple(cv.shape) == (1, 1, 3, 2, 4), f"got {tuple(cv.shape)}")
      check("cost_volume finite", torch.isfinite(cv).all().item())
      check("cost_volume zero padding at invalid disparity", cv[0, 0, 1, 0, 0].item() == 0.0)
      check("cost_volume shifted product semantics", cv[0, 0, 2, 0, 3].item() == left[0, 0, 0, 3].item() * right[0, 0, 0, 1].item())
    else:
      skip_checks(4, "cost_volume returned None")
  except Exception as exc:
    skip_checks(5, f"cost_volume raised {exc!r}")

  try:
    left = torch.arange(1, 25, dtype=torch.float32).view(1, 2, 3, 4)
    right = torch.flip(left, dims=[-1])
    dcv = dot_product_cost_volume(left, right, 3, False)
    module_out = DotProductCostVolume(3)(left, right)
    check("dot_product_cost_volume output not None", dcv is not None)
    if dcv is not None:
      check("dot_product_cost_volume shape", tuple(dcv.shape) == (1, 3, 3, 4), f"got {tuple(dcv.shape)}")
      check("dot_product_cost_volume finite", torch.isfinite(dcv).all().item())
      expected = (left[:, :, :, 2:] * right[:, :, :, :2]).mean(dim=1)
      check("dot_product_cost_volume shifted mean semantics", torch.allclose(dcv[:, 2, :, 2:], expected))
      check("DotProductCostVolume module delegates to scripted function", torch.allclose(module_out, dcv))
    else:
      skip_checks(4, "dot_product_cost_volume returned None")
  except Exception as exc:
    skip_checks(5, f"dot_product_cost_volume raised {exc!r}")

  try:
    cost = torch.tensor([[[[4., 1.], [0., 3.]],
                          [[2., 2.], [1., 1.]],
                          [[0., 4.], [3., 0.]]]])
    disp = soft_argmin(cost)
    module_disp = SoftArgmin()(cost)
    check("soft_argmin output not None", disp is not None)
    if disp is not None:
      check("soft_argmin shape", tuple(disp.shape) == (1, 1, 2, 2), f"got {tuple(disp.shape)}")
      check("soft_argmin finite", torch.isfinite(disp).all().item())
      check("soft_argmin values stay in disparity range", (disp.min().item() >= 0.0 and disp.max().item() <= 2.0))
      check("SoftArgmin module delegates to function", torch.allclose(module_disp, disp))
    else:
      skip_checks(4, "soft_argmin returned None")
  except Exception as exc:
    skip_checks(5, f"soft_argmin raised {exc!r}")

  try:
    net = NetFactory()
    x = net.input(in_dim=3, stride=1, activated=True)
    skip = net.downscale(x, 4)
    low = net.downscale(skip, 4)
    y = net.layer(low, out_dim=4, rate=1)
    up = net.upsample(y, skip, out_dim=4)
    net.tag(up, "p")
    baked = net.bake()
    data = torch.randn(2, 3, 16, 16)
    outputs = baked(data)
    check("NetFactory forward output not None", outputs is not None)
    if outputs is not None:
      check("NetFactory returns tagged dictionary", isinstance(outputs, dict) and "p" in outputs)
      check("NetFactory tagged feature shape", tuple(outputs["p"].shape) == (2, 4, 8, 8), f"got {tuple(outputs['p'].shape)}")
      check("NetFactory tagged feature finite", torch.isfinite(outputs["p"]).all().item())
      check("NetFactory retains skip connections", len(baked.skips) >= 2)
    else:
      skip_checks(4, "NetFactory returned None")
  except Exception as exc:
    skip_checks(5, f"NetFactory raised {exc!r}")

  try:
    hparams = HParams()
    stereo = HdrnAlphaStereo(hparams).eval()
    left = torch.randn(2, 3, 32, 32)
    right = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
      small_disp = stereo(left, right)
    check("HdrnAlphaStereo output not None", small_disp is not None)
    if small_disp is not None:
      check("HdrnAlphaStereo disparity shape", tuple(small_disp.shape) == (2, 1, 8, 8), f"got {tuple(small_disp.shape)}")
      check("HdrnAlphaStereo disparity finite", torch.isfinite(small_disp).all().item())
      check("HdrnAlphaStereo disparity bounded by internal bins", (small_disp.min().item() >= 0.0 and small_disp.max().item() <= 3.0))
    else:
      skip_checks(3, "HdrnAlphaStereo returned None")
  except Exception as exc:
    skip_checks(4, f"HdrnAlphaStereo raised {exc!r}")

  try:
    hparams = HParams()
    backbone = StereoBackbone(hparams).eval()
    image = torch.randn(2, 6, 32, 32)
    with torch.no_grad():
      features = backbone(image, step=0)
    check("StereoBackbone output not None", features is not None)
    if features is not None:
      check("StereoBackbone output keys", set(["p2", "p3", "p4", "small_disp"]).issubset(features.keys()))
      check("StereoBackbone p2 shape", tuple(features["p2"].shape) == (2, 64, 8, 8), f"got {tuple(features['p2'].shape)}")
      check("StereoBackbone p3 shape", tuple(features["p3"].shape) == (2, 64, 4, 4), f"got {tuple(features['p3'].shape)}")
      check("StereoBackbone p4 shape", tuple(features["p4"].shape) == (2, 64, 2, 2), f"got {tuple(features['p4'].shape)}")
      check("StereoBackbone outputs finite", all(torch.isfinite(features[k]).all().item() for k in ["p2", "p3", "p4", "small_disp"]))
    else:
      skip_checks(5, "StereoBackbone returned None")
  except Exception as exc:
    skip_checks(6, f"StereoBackbone raised {exc!r}")

  try:
    input_shape_4x = {
        "p2": ShapeSpec(channels=64, stride=4),
        "p3": ShapeSpec(channels=64, stride=8),
        "p4": ShapeSpec(channels=64, stride=16),
    }
    input_shape_8x = {
        "p3": ShapeSpec(channels=64, stride=8),
        "p4": ShapeSpec(channels=64, stride=16),
    }
    features = {
        "p2": torch.randn(2, 64, 8, 8),
        "p3": torch.randn(2, 64, 4, 4),
        "p4": torch.randn(2, 64, 2, 2),
    }
    sem_head = SemSegFPNHead(input_shape_4x, num_classes=5, model_norm="GN", num_filters_scale=4).eval()
    pose_head = PoseFPNHead(input_shape_8x, num_classes=16, model_norm="GN", num_filters_scale=4).eval()
    with torch.no_grad():
      sem = sem_head(features)
      pose = pose_head(features)
    check("SemSegFPNHead output not None", sem is not None)
    if sem is not None:
      check("SemSegFPNHead output shape", tuple(sem.shape) == (2, 5, 32, 32), f"got {tuple(sem.shape)}")
      check("SemSegFPNHead output finite", torch.isfinite(sem).all().item())
    else:
      skip_checks(2, "SemSegFPNHead returned None")
    check("PoseFPNHead output not None", pose is not None)
    if pose is not None:
      check("PoseFPNHead output shape", tuple(pose.shape) == (2, 16, 4, 4), f"got {tuple(pose.shape)}")
      check("PoseFPNHead output finite", torch.isfinite(pose).all().item())
    else:
      skip_checks(2, "PoseFPNHead returned None")
  except Exception as exc:
    skip_checks(6, f"FPN heads raised {exc!r}")

  try:
    hparams = HParams()
    model = PanopticNet(hparams).eval()
    image = torch.randn(2, 6, 32, 32)
    with torch.no_grad():
      outputs = model(image, step=0)
    check("PanopticNet output not None", outputs is not None)
    if outputs is not None:
      seg_output, depth_output, small_depth_output, pose_output, box_output, keypoint_output = outputs
      check("PanopticNet returns six outputs", len(outputs) == 6)
      check("PanopticNet segmentation shape", tuple(seg_output.seg_pred.shape) == (2, 5, 32, 32), f"got {tuple(seg_output.seg_pred.shape)}")
      check("PanopticNet refined depth shape", tuple(depth_output.depth_pred.shape) == (2, 32, 32), f"got {tuple(depth_output.depth_pred.shape)}")
      check("PanopticNet small disparity shape", tuple(small_depth_output.depth_pred.shape) == (2, 8, 8), f"got {tuple(small_depth_output.depth_pred.shape)}")
      check("PanopticNet OBB heatmap shape", tuple(pose_output.heatmap.shape) == (2, 32, 32), f"got {tuple(pose_output.heatmap.shape)}")
      check("PanopticNet OBB vertex/cov shapes", tuple(pose_output.vertex_field.shape) == (2, 16, 4, 4) and tuple(pose_output.cov_field.shape) == (2, 6, 4, 4))
      finite = (
          torch.isfinite(seg_output.seg_pred).all()
          and torch.isfinite(depth_output.depth_pred).all()
          and torch.isfinite(small_depth_output.depth_pred).all()
          and torch.isfinite(pose_output.heatmap).all()
          and torch.isfinite(pose_output.vertex_field).all()
          and torch.isfinite(pose_output.cov_field).all()
      )
      check("PanopticNet outputs finite tensors", finite.item())
      check("PanopticNet unused heads stay None", box_output is None and keypoint_output is None)
    else:
      skip_checks(8, "PanopticNet returned None")
  except Exception as exc:
    skip_checks(9, f"PanopticNet raised {exc!r}")

  print("=" * 70)
  print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
  print("=" * 70)
  if failed != 0:
    raise SystemExit(1)
