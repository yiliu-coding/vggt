"""
Minimal spconv shim for running PointTransformerV3 on CPU without real spconv.

Provides the minimal interface needed by Pointcept's PTv3:
- SparseConvTensor: data container
- SubMConv3d: loads real checkpoint weights, applies center-element linear transform
- modules.is_spconv_module(): for PointSequential dispatch
"""

import torch
import torch.nn as nn


class SparseConvTensor:
    """Minimal SparseConvTensor that stores features and indices."""

    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    def replace_feature(self, new_features):
        return SparseConvTensor(
            features=new_features,
            indices=self.indices,
            spatial_shape=self.spatial_shape,
            batch_size=self.batch_size,
        )


class SubMConv3d(nn.Module):
    """CPU SubMConv3d shim that uses center kernel element as a linear transform.

    Real spconv SubMConv3d has weight shape [out_ch, k, k, k, in_ch].
    We store the full weight for checkpoint loading, but only use the center
    element [out_ch, in_ch] for the forward pass (equivalent to 1x1x1 conv).
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, indice_key=None, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        # Match real spconv weight shape: [out_ch, k, k, k, in_ch]
        self.weight = nn.Parameter(torch.zeros(out_channels, *kernel_size, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        # Initialize center element as identity-ish
        k = kernel_size[0]
        c = k // 2
        nn.init.kaiming_uniform_(self.weight[:, c, c, c, :].data)

    def forward(self, x):
        if isinstance(x, SparseConvTensor):
            # Use center kernel element as linear transform
            k = self.kernel_size[0]
            c = k // 2
            center_weight = self.weight[:, c, c, c, :]  # [out_ch, in_ch]
            new_features = x.features @ center_weight.t()
            if self.bias is not None:
                new_features = new_features + self.bias
            return x.replace_feature(new_features)
        return x


class _Modules:
    """Stub for spconv.modules."""

    @staticmethod
    def is_spconv_module(module):
        return isinstance(module, SubMConv3d)


# Module-level attributes to mimic spconv.pytorch
modules = _Modules()
