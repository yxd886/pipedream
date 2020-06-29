# Training GPT-2

`launch.py` needs to be copied into the `gpt2` sub-directory to run these
commands.

## Data

We run throughput experiments using the WikiText-2 dataset, which can be
downloaded [here](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).

## Throughput Experiments

To run 200 iterations of training for a GPT-2 model across 8 8-GPU servers
without the above driver script, a command line of the following form can be used,

```bash
python -m launch --nnodes 8 \
        --node_rank 0 \
        --nproc_per_node=8 \
        main_with_runtime.py \
        --master_addr MASTER_ADDR \
        --module models.depth=8 \
        --train_batch_size 2 \
        --train_data_file TRAINING_DATA_FILE \
        --do_train \
        --num_minibatches 200 \
        --gradient_accumulation_steps 1 \
        --config_path conf.json
```

This needs to be run with appropriate arguments (`--node_rank`) on each
server. The model run is specified under `gpt2/depth=8`.
The PipeDream-2BW configuration used is specified using a `conf.json` file (example
pasted below).

```bash
{'module_to_stage_map': [0, 1, 2, 3, 4, 5, 6, 7, 7], 'stage_to_rank_map': {'0': [0, 1, 2, 3, 4, 5, 6, 7], '1': [8, 9, 10, 11, 12, 13, 14, 15], '2': [16, 17, 18, 19, 20, 21, 22, 23], '3': [24, 25, 26, 27, 28, 29, 30, 31], '4': [32, 33, 34, 35, 36, 37, 38, 39], '5': [40, 41, 42, 43, 44, 45, 46, 47], '6': [48, 49, 50, 51, 52, 53, 54, 55], '7': [56, 57, 58, 59, 60, 61, 62, 63]}}
```

We assume that all necessary software dependencies
are available (for example, the above command line can be run using the Docker
container built using instructions in the top-level README).
