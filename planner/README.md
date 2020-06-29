# Planner

This directory contains an planner that can determine how to partition the input
model over the available workers.

The planner has the following command line arguments,

```bash
python planner.py -h
usage: planner.py [-h] -b NUM_BLOCKS -n NUM_WORKERS [-m MEMORY_CAPACITY] -t
                    COMPUTATION_TIME_PER_BLOCK -p NUM_PARAMETERS_PER_BLOCK -a
                    NUM_ACTIVATIONS_PER_BLOCK -o OUTPUT_ACTIVATION_SIZE

Run the PipeDream-2BW planner

optional arguments:
  -h, --help            show this help message and exit
  -b NUM_BLOCKS, --num_blocks NUM_BLOCKS
                        Number of blocks in model
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        Number of workers
  -m MEMORY_CAPACITY, --memory_capacity MEMORY_CAPACITY
                        Amount of memory available on each machine
  -t COMPUTATION_TIME_PER_BLOCK, --computation_time_per_block COMPUTATION_TIME_PER_BLOCK
                        Computation time per block
  -p NUM_PARAMETERS_PER_BLOCK, --num_parameters_per_block NUM_PARAMETERS_PER_BLOCK
                        Number of weight parameters per block
  -a NUM_ACTIVATIONS_PER_BLOCK, --num_activations_per_block NUM_ACTIVATIONS_PER_BLOCK
                        Number of activations per block
  -o OUTPUT_ACTIVATION_SIZE, --output_activation_size OUTPUT_ACTIVATION_SIZE
                        Output activation size
```
