# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .start_stage import StartingStage
from .intermediate_stage import IntermediateStage
from .end_stage import EndingStage

def arch():
    return "gpt2"

def model(config, criterion):
    stage0 = StartingStage(config)
    stage1 = IntermediateStage(config)
    stage2 = IntermediateStage(config)
    stage3 = IntermediateStage(config)
    stage4 = IntermediateStage(config)
    stage5 = IntermediateStage(config)
    stage6 = IntermediateStage(config)
    stage7 = IntermediateStage(config)
    stage8 = IntermediateStage(config)
    stage9 = IntermediateStage(config)
    stage10 = IntermediateStage(config)
    stage11 = IntermediateStage(config)
    stage12 = IntermediateStage(config)
    stage13 = IntermediateStage(config)
    stage14 = IntermediateStage(config)
    stage15 = EndingStage(config)
    return [
        (stage0, ["input_ids"], ["out0"]),
        (stage1, ["out0"], ["out1"]),
        (stage2, ["out1"], ["out2"]),
        (stage3, ["out2"], ["out3"]),
        (stage4, ["out3"], ["out4"]),
        (stage5, ["out4"], ["out5"]),
        (stage6, ["out5"], ["out6"]),
        (stage7, ["out6"], ["out7"]),
        (stage8, ["out7"], ["out8"]),
        (stage9, ["out8"], ["out9"]),
        (stage10, ["out9"], ["out10"]),
        (stage11, ["out10"], ["out11"]),
        (stage12, ["out11"], ["out12"]),
        (stage13, ["out12"], ["out13"]),
        (stage14, ["out13"], ["out14"]),
        (stage15, ["out14"], ["out15"]),
        (criterion, ["out15"], ["loss"])
    ]
