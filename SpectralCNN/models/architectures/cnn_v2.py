import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Type, Tuple
from typing import List, Optional, Dict, Union
import math
import logging

logger = logging.getLogger(__name__)


### --- Models --- ###
class Baseline1DCNN(nn.Module):
    """Dual-Head 1D-CNN for Feature Extraction and Classification."""

    def __init__(
        self,
        input_channels=1,
        num_filters=32,
        kernel_size=3,
        dropout=0.5,
        num_classes=10,
        num_convs=2,
        norm_type="bn",  # "bn" for BatchNorm, "gn" for GroupNorm
        gn_groups=None,  # Number of groups for GroupNorm; if None, defaults to min(32, out_channels)
        input_length: int = 4096,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.output_length = input_length // (2**num_convs)

        in_channels = input_channels
        for i in range(num_convs):
            out_channels = num_filters * (2**i)
            # convs
            conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=kernel_size // 2
            )
            self.convs.append(conv)

            # norm
            if norm_type.lower() == "bn":
                norm = nn.BatchNorm1d(out_channels)
            elif norm_type.lower() == "gn":
                norm = nn.GroupNorm(
                    gn_groups if gn_groups <= out_channels else 1, out_channels
                )
            else:
                norm = nn.Identity()
                # raise ValueError(f"Unknown norm_type: {norm_type}")

            self.norms.append(norm)
            in_channels = out_channels

        self.feat_len = in_channels * self.output_length
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # two heads
        self.head_hard = nn.Linear(self.feat_len, num_classes)
        self.head_soft = nn.Linear(self.feat_len, num_classes)
        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(conv.bias, 0)
        nn.init.normal_(self.head_hard.weight, 0, 0.01)
        nn.init.constant_(self.head_hard.bias, 0)
        nn.init.normal_(self.head_soft.weight, 0, 0.1)
        nn.init.constant_(self.head_soft.bias, 0.5)

    def forward(self, x):
        # spec: [B, 1, L]
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x)))
            x = F.max_pool1d(x, 2)

        # flatten to [B, feature_length]
        feats = x.flatten(start_dim=1)
        feats = self.dropout(feats)

        # logits: [B, num_classes]
        logits_hard = self.head_hard(feats)
        logits_soft = self.head_soft(feats)
        return feats, logits_hard, logits_soft

    def __repr__(self):
        return f"Baseline1DCNN(num_convs={len(self.convs)}, num_classes={self.head_hard.out_features}, feat_len={self.feat_len})"


##########################################################################################
##                              --- Attention CNN ---                                   ##
##########################################################################################
class ArcFace(nn.Module):
    """
    Implementation of the ArcFace layer.
    Reference: https://arxiv.org/abs/1801.07698
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # scale factor
        self.m = m  # angular margin

        # weight is the learnable center for each class on the hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # margin calc precomputation
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # normalize features and weights
        cosine = F.linear(
            F.normalize(input), F.normalize(self.weight)
        )  # [B, out_features]
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))

        # phi = cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # add margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # scale up

        return output


class SE1D(nn.Module):
    """
    A Squeeze-and-Excitation block for 1D data.
    This block adaptively recalibrates channel-wise feature responses by
    explicitly modeling interdependencies between channels.
    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for the intermediate channel size.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        # mid: intermediate channel size
        mid = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(x)
        return x * w


class ECA1D(nn.Module):
    """
    A Efficient Channel Attention block for 1D data.
    This block adaptively recalibrates channel-wise feature responses by
    explicitly modeling interdependencies between channels.
    Does not need Fully-connected layers.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gap(x)
        y = y.transpose(1, 2)  # [B,C,1] -> [B, 1, C]
        y = self.sigmoid(self.conv(y))
        y = y.transpose(1, 2)  # [B, 1, C] -> [B, C, 1]
        return x * y


class TemporalGate1D(nn.Module):
    """
    A Temporal Gate block for 1D data.
    This block adaptively recalibrates temporal feature responses by
    explicitly modeling interdependencies between time steps.
    CBAM-style temporal gate after pooling.
    Is a spatial/temporal attention module that can be used in combination with channel attention (SE, ECA).
    1D-Conv over length with sigmoid.
    """

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.sigmoid(y)
        return x * y


class BottleneckMHSA(nn.Module):
    """
    Bottleneck Multi-Head Self-Attention block for 1D data.
    Only once L is small (e.g., after 2-3 pools: 4096 -> 512/256)
    Pros: Richer Relations, BUT: more params and easier to overfit.
    Fit: use few heads, small d_model and only at bottleneck.
    """

    def __init__(self, channels, num_heads=2, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + y
        x = x + self.ff(self.ln2(x))
        x = x.transpose(1, 2)  # [B, L, C] -> [B, C, L]
        return x


class AttentionPooling1D(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention_scorer = nn.Linear(in_features, 1)

    def forward(self, x):
        # x shape: [Batch, Channels, Seq_Length]
        # transpose to [Batch, Seq_Length, Channels] for the linear layer
        x = x.transpose(1, 2)

        # compute scores for each time step
        # [B, L, C] -> [B, L, 1]
        attention_scores = self.attention_scorer(x)

        # turn scores into weights using softmax
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Compute the weighted average
        # (x * weights) -> [B, L, C] --sum(dim=1)--> [B, C]
        context_vector = torch.sum(x * attention_weights, dim=1)

        return context_vector


def make_norm(kind, channels, gn_groups=None):
    if kind.lower() == "bn":
        return nn.BatchNorm1d(channels)
    if kind.lower() == "gn":
        return nn.GroupNorm(gn_groups or min(32, channels), channels)
    if kind.lower() == "ln":
        return nn.GroupNorm(1, channels)  # GroupNorm with 1 group == LayerNorm
    return nn.Identity()  # No normalization


def get_activation(name: str) -> Type[nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


def get_attention_module(name: str, channels: int) -> Type[nn.Module]:
    """Helper to get attention module by name."""
    name = name.lower()
    if name == "se":
        return SE1D(channels, reduction=8)
    elif name == "eca":
        return ECA1D(channels)
    elif name == "temporal":
        return TemporalGate1D(channels)
    elif name == "hybrid":
        # hybrid combines ECA + TemporalGate, inspired by CBAM
        return nn.Sequential(ECA1D(channels), TemporalGate1D(channels))
    else:
        return nn.Identity()  # no attention


def get_feature_extractor(
    final_channels: int, attention_pooling: bool = False
) -> Type[nn.Module]:
    """Helper to get feature extractor module."""
    if attention_pooling:
        return AttentionPooling1D(in_features=final_channels)
    else:
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global Average Pooling
            nn.Flatten(),
        )


class DepthwiseSeparableConv1D(nn.Module):
    """
    A 1D depthwise separable convolution module.
    This is a building block for more efficient CNNs.
    Note:
    - Standard Conv: Parameters = C_in * C_out * kernel_size = 64 * 128 * 7 = 57,344
    - Separable Conv:
      - Depthwise: C_in * 1 * kernel_size = 64 * 1 * 7 = 448
      - Pointwise: C_in * C_out * 1 = 64 * 128 * 1 = 8,192
      - Total = 448 + 8,192 = 8,640 (--> 85% reduction in params for this single layer!)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # The 'groups' argument is the key to making this a depthwise convolution.
        # Each input channel gets its own separate filter.
        # spatial filtering step, doesn't mix information between channels
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,  # Bias is usually omitted in the depthwise conv
        )
        # The pointwise convolution is a simple 1x1 convolution that mixes the channels.
        # channel combination step, projects features from depthwise step into a new channel space
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,  # Bias can be included in the pointwise conv
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: str,
        attention_type: str = None,
        stride: int = 1,
        group_norm_groups: Optional[int] = None,
        use_depthwise: bool = True,  # whether to use depthwise separable convs or standard convs
        activation: str = "relu",
    ):
        super().__init__()
        self.use_depthwise = use_depthwise
        self.activation = get_activation(activation)
        conv_class = self._get_conv_class()
        padding = kernel_size // 2

        # --- prepare args ---
        conv1_args = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        conv2_args = dict(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        # --- prepare convolution type ---
        if not self.use_depthwise:
            # for standard conv
            conv1_args["bias"], conv2_args["bias"] = False, False
            conv1_args["padding"], conv2_args["padding"] = (
                kernel_size // 2,
                kernel_size // 2,
            )

        # --- MAIN PATH ---
        self.main_path = nn.Sequential(
            # first convolutional layer
            conv_class(**conv1_args),
            make_norm(norm_type, out_channels, group_norm_groups),
            self.activation,
            # second convolutional layer
            conv_class(**conv2_args),
            make_norm(norm_type, out_channels, group_norm_groups),
        )

        # --- Attention Module ---
        self.attn_module = get_attention_module(attention_type, out_channels)

        # skip connection path (residual), handles the 'x' in F(x) + x
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                make_norm(norm_type, out_channels, group_norm_groups),
            )

    def _get_conv_class(self) -> Type[nn.Module]:
        if self.use_depthwise:
            return DepthwiseSeparableConv1D
        else:
            return nn.Conv1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # final output is addition of main and skip paths
        main_out = self.main_path(x)  # F(x)

        # apply attention if it exists (Attn(F(x)))
        main_out = self.attn_module(main_out)

        # add skip connection
        out = main_out + self.skip_connection(x)  # Attn(F(x)) + x
        # final non-linearity activation
        out = self.activation(out)
        return out


# --- CNN builder: CNN + {none|se|eca|temporal} per block, optional MHSA at bottleneck ---
class Attention1DCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_convs: int = 3,
        dropout: float = 0.4,
        norm_type: str = "gn",
        kernel_size: int = 7,
        kernel_size_stem: int = 15,
        num_filters: int = 32,
        attention_type: str = "eca",
        activation: str = "relu",
        group_norm_groups: Optional[
            int
        ] = None,  # for GN layers if used. If None, defaults to min(32, out_channels)
        use_depthwise: bool = True,  # whether to use depthwise separable convs or standard convs
        verbose: bool = False,
        use_attention_pooling: bool = False,
    ):
        super().__init__()
        # store key parameters for use
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.attention_type = attention_type
        self.activation = get_activation(activation)
        self.group_norm_groups = group_norm_groups
        self.use_depthwise = use_depthwise
        self.use_attention_pooling = use_attention_pooling
        # for repr
        self.num_classes = num_classes
        self.num_stages = num_convs
        self.blocks_per_stage = 2  # fixed to 2 for now, can be made configurable later

        if verbose:
            logger.info(
                f"Attention1DCNN: norm={norm_type}, attn={attention_type}, use_depthwise={use_depthwise}, activation={activation}, gn_groups={group_norm_groups}, num_convs={num_convs}, num_filters={num_filters}, dropout={dropout}, num_classes={num_classes}, kernel_size={kernel_size}."
            )

        # --- precise Stem, not downsampling, but extracting features ---
        self.stem = nn.Sequential(
            nn.Conv1d(
                1,
                num_filters,
                kernel_size=kernel_size_stem,
                padding=kernel_size_stem // 2,
                stride=1,
                bias=False,
            ),
            make_norm(norm_type, num_filters),
            self.activation,
        )

        # --- Body: series of res blocks that gradually downsample ---
        self.body = self._make_body(
            in_channels=num_filters,
            num_stages=num_convs,
            blocks_per_stage=self.blocks_per_stage,
        )

        # --- Head: Pooling + Classifier ---
        final_channels = num_filters * (2**num_convs)  # 32 * 8 = 256
        ### Shared part: produces final feature vector
        self.feature_extractor = get_feature_extractor(
            final_channels=final_channels, attention_pooling=self.use_attention_pooling
        )
        ### Heads for "hard" and "soft" labels
        self.head_hard = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_channels, num_classes),
        )
        self.head_soft = nn.Sequential(
            nn.Dropout(dropout),  # each head can have its own dropout
            nn.Linear(final_channels, num_classes),
        )

    def _make_body(
        self, in_channels: int, num_stages: int, blocks_per_stage: int
    ) -> nn.Sequential:
        layers = []

        # looping through each stage of the network
        current_channels = in_channels
        for i in range(num_stages):
            # first block of each stage handles downsampling with stride=2
            # except for the very first stage, which takes the stem output
            stride = 2 if i > 0 else 1

            # doubling the number of channels at each stage
            next_channels = current_channels * 2

            layers.append(
                ResidualBlock1D(
                    current_channels,
                    next_channels,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    attention_type=self.attention_type,
                    stride=stride,
                    group_norm_groups=self.group_norm_groups,
                    use_depthwise=self.use_depthwise,
                )
            )  # downsampling block

            # for rest of blocks for this stage, no downsampling
            for _ in range(1, blocks_per_stage):
                layers.append(
                    ResidualBlock1D(
                        next_channels,
                        next_channels,
                        kernel_size=self.kernel_size,
                        norm_type=self.norm_type,
                        attention_type=self.attention_type,
                        stride=1,  # notice stride=1
                    )
                )

            current_channels = next_channels  # output of this stage is input to next

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        feats = self.feature_extractor(x)

        out_hard = self.head_hard(feats)
        out_soft = self.head_soft(feats)
        return feats, out_hard, out_soft

    def __repr__(self):
        return f"Attention1DCNN(num_convs={len(self.body)}, num_classes={self.num_classes}, feat_len={self.head_hard[-1].in_features}, num_stages={self.num_stages}, blocks_per_stage={self.blocks_per_stage})"
