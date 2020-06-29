# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .start_stage import StartingStage
from .intermediate_stage import IntermediateStage
from .end_stage import EndingStage

def arch():
    return "bert"

def model(config, criterion):
    stage0 = StartingStage(config)
    stage1 = IntermediateStage(config)
    stage2 = IntermediateStage(config)
    stage3 = EndingStage(config, stage0.embedding_layer.word_embeddings.weight)
    return [
        (stage0, ["input0", "input1", "input2"], ["out1", "out0"]),
        (stage1, ["out1", "out0"], ["out3", "out2"]),
        (stage2, ["out3", "out2"], ["out5", "out4"]),
        (stage3, ["out5", "out4"], ["out6"]),
        (criterion, ["out6"], ["loss"])
    ]
