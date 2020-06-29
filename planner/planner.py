# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse


def bandwidth_depth(d):
    return 1000000000

def bandwidth_width(w):
    if w <= 8:
        return 100000000000
    return 1000000000

def performance_cost_fn(num_blocks, computation_time_per_block, num_parameters_per_block,
                        output_activation_size, w, d):
    computation_time = computation_time_per_block * (num_blocks // d)
    num_parameters = num_parameters_per_block * (num_blocks // d)

    return max(computation_time,
               (2 * output_activation_size) / bandwidth_depth(d),
               (2 * (w-1) * num_parameters) / bandwidth_width(w)) / w


def memory_cost_fn(num_blocks, num_parameters_per_block, num_activations_per_block, w, d):
    if num_blocks < d:
        num_blocks += d
    num_parameters = num_parameters_per_block * (num_blocks // d)
    num_activations = num_activations_per_block * (num_blocks // d)
    return ((num_parameters * 2) + (num_activations * d)) * 4.0


def partition(num_blocks, num_workers, memory_capacity, performance_cost_fn,
              memory_cost_fn):
    best_performance_cost, best_w, best_d = None, None, None
    for w in range(1, num_workers + 1):
        for d in range(1, (num_workers // w) + 1):
            if w * d != num_workers:
                continue
            print(w, d, memory_cost_fn(w, d))
            if memory_cost_fn(w, d) > memory_capacity:
                continue
            performance_cost = performance_cost_fn(w, d)
            if (best_performance_cost is None or \
                best_performance_cost > performance_cost):
                best_performance_cost = performance_cost
                best_w = w
                best_d = d
    print(best_w, best_d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run the PipeDream-2BW optimizer")
    )
    parser.add_argument('-b', "--num_blocks", required=True, type=int,
                        help="Number of blocks in model")
    parser.add_argument('-n', "--num_workers", required=True, type=int,
                        help="Number of workers")
    parser.add_argument('-m', "--memory_capacity", default=16000000000, type=float,
                        help="Amount of memory available on each machine")
    parser.add_argument('-t', "--computation_time_per_block", required=True,
                        type=float, help="Computation time per block")
    parser.add_argument('-p', "--num_parameters_per_block", required=True,
                        type=int, help="Number of weight parameters per block")
    parser.add_argument('-a', "--num_activations_per_block", required=True,
                        type=int, help="Number of activations per block")
    parser.add_argument('-o', "--output_activation_size", required=True,
                        type=int, help="Output activation size")
    args = parser.parse_args()

    partition(
        args.num_blocks, args.num_workers, args.memory_capacity,
        lambda w, d: performance_cost_fn(args.num_blocks,
                                         args.computation_time_per_block,
                                         args.num_parameters_per_block,
                                         args.output_activation_size,
                                         w, d),
        lambda w, d: memory_cost_fn(args.num_blocks,
                                    args.num_parameters_per_block,
                                    args.num_activations_per_block,
                                    w, d)
    )
