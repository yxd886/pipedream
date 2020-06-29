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
    stage7 = EndingStage(config)
    return [
        (stage0, ["input_ids"], ["out0"]),
        (stage1, ["out0"], ["out1"]),
        (stage2, ["out1"], ["out2"]),
        (stage3, ["out2"], ["out3"]),
        (stage4, ["out3"], ["out4"]),
        (stage5, ["out4"], ["out5"]),
        (stage6, ["out5"], ["out6"]),
        (stage7, ["out6"], ["out7"]),
        (criterion, ["out7"], ["loss"])
    ]
