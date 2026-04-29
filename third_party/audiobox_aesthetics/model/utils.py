# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn


def create_mlp_block(input_dim, output_dim, num_layer, act_fn, layer_norm, dropout=0):
    proj_layer = []
    for ii in range(num_layer):
        if ii == num_layer - 1:
            proj_layer.append(nn.Linear(input_dim, output_dim))
        else:
            proj_layer.append(nn.Linear(input_dim, input_dim))
            if layer_norm:
                proj_layer.append(nn.LayerNorm(normalized_shape=(input_dim)))
            if act_fn == "gelu":
                proj_layer.append(nn.GELU())
            else:
                raise ValueError()
            if dropout != 0:
                proj_layer.append(nn.Dropout(p=dropout))
    return proj_layer
