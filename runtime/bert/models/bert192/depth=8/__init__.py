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
    stage3 = IntermediateStage(config)
    stage4 = IntermediateStage(config)
    stage5 = IntermediateStage(config)
    stage6 = IntermediateStage(config)
    stage7 = EndingStage(config, stage0.embedding_layer.word_embeddings.weight)
    return [
        (stage0, ["input0", "input1", "input2"], ["out1", "out0"]),
        (stage1, ["out1", "out0"], ["out3", "out2"]),
        (stage2, ["out3", "out2"], ["out5", "out4"]),
        (stage3, ["out5", "out4"], ["out7", "out6"]),
        (stage4, ["out7", "out6"], ["out9", "out8"]),
        (stage5, ["out9", "out8"], ["out11", "out10"]),
        (stage6, ["out11", "out10"], ["out13", "out12"]),
        (stage7, ["out13", "out12"], ["out14"]),
        (criterion, ["out14"], ["loss"])
    ]
