# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stage0 import Stage0
from .stage1 import Stage1

def arch():
    return "bert"

def model(config, criterion):
    stage0 = Stage0(config)
    stage1 = Stage1(config, stage0.embedding_layer.word_embeddings.weight)
    return [
        (stage0, ["input0", "input1", "input2"], ["out1", "out0"]),
        (stage1, ["out1", "out0"], ["out2"]),
        (criterion, ["out2"], ["loss"])
    ]