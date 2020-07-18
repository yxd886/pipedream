# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import contextlib
import glob
import logging
import json
import os
import pickle
import random
import re
import shutil
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import sys
sys.path.append("..")
import runtime
from torch.optim.optimizer import required
from optimizer_with_stashing import OptimizerWithStashing
from optimizer_with_stashing_and_aggregation import OptimizerWithStashingAndAggregation
from ptflops import get_model_complexity_info


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper methods.

def save_checkpoint(state, checkpoint_dir, stage, epoch):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(checkpoint_dir,
                                        "checkpoint.%d.pth.tar.epoch.%d" % (stage, epoch))
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)


class AdamWWithStashingAndAggregation(OptimizerWithStashingAndAggregation):
    """
    AdamW optimizer with weight stashing and aggregation (to reduce memory overhead).
    """
    def __init__(self, modules, master_parameters, num_stages, update_interval,
                 lr=required, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0,
                 correct_bias=True, verbose_freq=0):
        super(AdamWWithStashingAndAggregation, self).__init__(
            optim_name='AdamW', modules=modules,
            master_parameters=master_parameters, num_stages=num_stages,
            update_interval=update_interval,
            verbose_freq=verbose_freq, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, correct_bias=correct_bias,
            base_optimizer_cls=AdamW
        )

class AdamWWithStashing(OptimizerWithStashing):
    """
    AdamW optimizer with weight stashing.
    """
    def __init__(self, modules, master_parameters, num_versions,
                 lr=required, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0,
                 correct_bias=True, verbose_freq=0):
        super(AdamWWithStashing, self).__init__(
            optim_name='AdamW',
            modules=modules, master_parameters=master_parameters,
            model_parameters=None, loss_scale=1.0,
            num_versions=num_versions, verbose_freq=verbose_freq,
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, correct_bias=correct_bias,
            base_optimizer_cls=AdamW
        )

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "gpt2_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, configuration_maps, train_dataset,
          r, tokenizer) -> Tuple[int, float]:
    """ Train the model """

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = None
        distributed_sampler = False
        if configuration_maps['stage_to_rank_map'] is not None:
            num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
            if num_ranks_in_first_stage > 1:
                train_sampler = DistributedSampler(
                train_dataset, num_replicas=num_ranks_in_first_stage,
                rank=args.rank)
                distributed_sampler = True

        if not distributed_sampler:
            train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.num_minibatches is not None:
        t_total = args.num_minibatches
        args.num_train_epochs = args.num_minibatches // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # TODO: Figure out how to call this method.
    # model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in r.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in r.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.pipedream:
        optimizer = AdamWWithStashing(r.modules(),
                                      optimizer_grouped_parameters,
                                      num_versions=r.num_warmup_minibatches+1,
                                      lr=args.learning_rate,
                                      eps=args.adam_epsilon,
                                      verbose_freq=args.verbose_frequency)
    elif args.gpipe or args.no_input_pipelining:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon)
    else:
        optimizer = AdamWWithStashingAndAggregation(
            r.modules(),
            optimizer_grouped_parameters,
            num_stages=r.num_stages,
            update_interval=r.update_interval,
            lr=args.learning_rate,
            eps=args.adam_epsilon)

    if not args.gpipe and not args.no_input_pipelining:
        optimizer.initialize_queue()

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    set_seed(args)
    if args.num_minibatches is not None:
        args.num_train_epochs = 1
    for epoch in range(int(args.num_train_epochs)):
        epoch_start_time = time.time()

        update_interval = r.update_interval

        n = len(train_dataloader)
        if args.num_minibatches is not None:
            n = args.num_minibatches
        # Number of iterations should be multiple of update_interval.
        n = ((n // update_interval)) * update_interval

        if r.is_first_stage():
            input_source = InputSource(train_dataloader, r.parameters())
            r.set_input_source(input_source)

        if args.gpipe:
            r.run_training_loop_with_flushes(n, optimizer, args.recompute_step)
        else:
            recompute_step = args.recompute_step and r.stage is not None and \
                (r.stage != (r.num_stages - 1))
            r.run_training_loop(n, optimizer, recompute_step,
                                args.no_input_pipelining)

        print("Epoch %d (%d iterations): %.3f seconds" % (
            epoch, n, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" % (
            epoch_start_time, time.time()))

        # Barrier after completing iterations to wait for other ranks to finish.
        if args.local_rank != -1:
            import torch.distributed as dist; dist.barrier()


def evaluate(args, model, tokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched).

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    # Note that DistributedSampler samples randomly.

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in eval_dataloader:
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            lm_loss = model(inputs, labels=labels)
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return result


class InputSource(runtime.InputSourceBase):
    def __init__(self, dataloader, parameters):
        self.loader_iter = iter(dataloader)
        self.dtype = next(parameters).dtype

    def get_inputs(self):
        input_tensors = {}

        batch = next(self.loader_iter)
        batch = batch.to("cuda")

        input_tensors["input_ids"] = batch
        input_tensors["labels"] = batch

        return input_tensors


def main():
    parser = argparse.ArgumentParser()

    # Required parameters.
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True,
        help="The input training data file (a text file)."
    )

    # Other parameters.
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true",
        help="Whether to continue from latest checkpoint in output_dir"
    )

    # PipeDream arguments.
    parser.add_argument('--module', '-m', required=True,
                        help='name of module that contains model and tensor_shapes definition')
    parser.add_argument('--master_addr', default=None, type=str,
                        help="IP address of master (machine with rank 0)")
    parser.add_argument('--config_path', default=None, type=str,
                        help="Path of configuration file")
    parser.add_argument('--no_input_pipelining', action='store_true',
                        help="No pipelining of inputs")
    parser.add_argument('--rank', default=None, type=int,
                        help="Rank of worker")
    parser.add_argument('--num_ranks_in_server', default=1, type=int,
                        help="number of gpus per machine")
    parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                        help="Log verbose information")
    # Recompute tensors from forward pass, instead of saving them.
    parser.add_argument('--recompute_step', action='store_true',
                        help='Recompute tensors in backward pass')
    # PipeDream-style execution.
    parser.add_argument('--pipedream', action='store_true',
                        help='Use PipeDream-style weight updates with worse memory efficiency')
    # GPipe-style execution.
    parser.add_argument('--gpipe', action='store_true',
                        help='Use GPipe-style weight updates')

    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument(
        "--train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument('--num_minibatches', default=None, type=int,
                        help="Number of minibatches to run.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    # Setup CUDA, GPU & distributed training.
    local_rank = 0
    if args.local_rank != -1:
        local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.device = device

    logger.warning(
        "Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed.
    set_seed(args)

    config = GPT2Config(n_layer=32, n_embd=1280, n_head=20)
    tokenizer = AutoTokenizer.from_pretrained(
        'gpt2', config=config, cache_dir=args.cache_dir)

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model.
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    import importlib
    module = importlib.import_module(args.module)
    criterion = torch.nn.CrossEntropyLoss()
    arch = module.arch()
    model = module.model(config, criterion)

    input_size = [args.train_batch_size, 1024]
    training_tensor_shapes = {"input_ids": input_size, "labels": input_size}
    dtypes = {"input_ids": torch.int64, "labels": torch.int64}
    inputs_module_destinations = {"input_ids": 0, "labels": 0}
    target_tensor_names = {"labels"}

    total_flops = 0
    total_params = 0
    total_trainable_params = 0
    for module_id, (stage_module_fn, inputs, outputs) in enumerate(model[:-1]):  # Skip last layer (loss).
        input_tensors = []
        for module_input in inputs:
            if module_input in inputs_module_destinations:
                inputs_module_destinations[module_input] = module_id

            input_tensor = torch.ones(tuple(training_tensor_shapes[module_input]),
                                      dtype=dtypes[module_input]).cuda()
            input_tensors.append(input_tensor)
        stage_module = stage_module_fn()
        stage_module.cuda()
        # PyTorch should not maintain metadata for a backward pass on
        # synthetic inputs. Without the following line, the runtime is
        # as much as 1.5x slower in a full DP configuration.
        with torch.no_grad():
            output_tensors = stage_module(*tuple(input_tensors))
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype
        model_complexity_info = get_model_complexity_info(stage_module, tuple(input_tensors), print_per_layer_stat=False)
        total_flops += float(model_complexity_info[0].split(" ")[0])
        total_params += sum(p.numel() for p in stage_module.parameters())
        total_trainable_params += sum(p.numel() for p in stage_module.parameters() if p.requires_grad)
        del stage_module
    print("Total number of floating point operations: %.2f * 10**9" % (
        total_flops * args.train_batch_size * 6))
    print("Total number of parameters: %d" % total_params)
    print("Total number of trainable parameters: %d" % total_trainable_params)

    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)

    r = runtime.StageRuntime(
        model=model,
        fp16=args.fp16, loss_scale=1.0,  # fp32 training, so disable loss scaling.
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr, rank=args.rank,
        local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.GPT2)
    update_interval = r.num_stages
    if args.pipedream:
        update_interval = 1
    update_interval *= args.gradient_accumulation_steps
    r.update_interval = update_interval
    r.vocab_size = config.vocab_size
    r.cuda()

    # Stage needed to determine if current stage is the first stage.
    # num_stages needed to determine if current stage is the last stage.
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining.
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks
    if not r.is_first_stage():
        args.synthetic_data = True

    r.initialize_distributed_backend()

    # Training.
    if args.do_train:
        # Barrier to make sure only the first process in distributed training process
        # the dataset, and the others will use the cache.
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, configuration_maps, train_dataset, r, tokenizer)

    # Evaluation.
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        result = evaluate(args, model, tokenizer)
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)

    return results


if __name__ == "__main__":
    main()
