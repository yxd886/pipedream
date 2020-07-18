# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .start_stage import StartingStage
from .intermediate_stage import IntermediateStage
from .end_stage import EndingStage
from transformers.modeling import BertEmbeddings

def arch():
    return "bert"

def model(config, criterion):
    return [
        (lambda: StartingStage(config), ["input0", "input1", "input2"], ["out1", "out0"]),
        (lambda: IntermediateStage(config), ["out1", "out0"], ["out3", "out2"]),
        (lambda: IntermediateStage(config), ["out3", "out2"], ["out5", "out4"]),
        (lambda: IntermediateStage(config), ["out5", "out4"], ["out7", "out6"]),
        (lambda: IntermediateStage(config), ["out7", "out6"], ["out9", "out8"]),
        (lambda: IntermediateStage(config), ["out9", "out8"], ["out11", "out10"]),
        (lambda: IntermediateStage(config), ["out11", "out10"], ["out13", "out12"]),
        (lambda: IntermediateStage(config), ["out13", "out12"], ["out15", "out14"]),
        (lambda: IntermediateStage(config), ["out15", "out14"], ["out17", "out16"]),
        (lambda: IntermediateStage(config), ["out17", "out16"], ["out19", "out18"]),
        (lambda: IntermediateStage(config), ["out19", "out18"], ["out21", "out20"]),
        (lambda: IntermediateStage(config), ["out21", "out20"], ["out23", "out22"]),
        (lambda: IntermediateStage(config), ["out23", "out22"], ["out25", "out24"]),
        (lambda: IntermediateStage(config), ["out25", "out24"], ["out27", "out26"]),
        (lambda: IntermediateStage(config), ["out27", "out26"], ["out29", "out28"]),
        (lambda: EndingStage(config, BertEmbeddings(config).word_embeddings.weight), ["out29", "out28"], ["out30"]),
        (lambda: criterion, ["out30"], ["loss"])
    ]
