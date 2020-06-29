# Pre-training and fine-tuning BERT models

`launch.py` needs to be copied into the `bert` sub-directory to run these
commands.

## Data

We run throughput experiments using a subset of the Wikipedia dataset.
Instructions to download the Wikipedia dataset needed for pre-training
are [here](https://github.com/microsoft/AzureML-BERT/blob/master/pretrain/PyTorch/notebooks/BERT_Pretrain.ipynb).
To download the GLUE dataset needed for fine-tuning the pre-trained model,
use [this script](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py).

## Throughput Experiments

To run 200 iterations of training for a BERT-24 model across 8 8-GPU servers
without the above driver script, a command line of the following form can be used,

```bash
python -m launch --nnodes 8 \
        --node_rank 0 \
        --nproc_per_node=8 \
        main_with_runtime.py \
        --master_addr MASTER_ADDR \
        --module models.bert24.depth=8 \
        --max_seq_length 128 \
        --train_batch_size 16 \
        --train_path TRAINING_DATA_PATH \
        --bert_config_path configs/bert_config_bert-large-uncased.json \
        --vocab_path TRAINING_DATA_PATH/vocab.txt \
        --do_train \
        --on_memory \
        --do_lower_case \
        --num_minibatches 200 \
        --gradient_accumulation_steps 1 \
        --config_path conf.json
```

The PipeDream-2BW configuration used is specified using a `conf.json` file (example
pasted below).

```bash
{'module_to_stage_map': [0, 1, 2, 3, 4, 5, 6, 7, 7], 'stage_to_rank_map': {'0': [0, 1, 2, 3, 4, 5, 6, 7], '1': [8, 9, 10, 11, 12, 13, 14, 15], '2': [16, 17, 18, 19, 20, 21, 22, 23], '3': [24, 25, 26, 27, 28, 29, 30, 31], '4': [32, 33, 34, 35, 36, 37, 38, 39], '5': [40, 41, 42, 43, 44, 45, 46, 47], '6': [48, 49, 50, 51, 52, 53, 54, 55], '7': [56, 57, 58, 59, 60, 61, 62, 63]}}
```

We assume that all necessary software dependencies
are available (for example, the above command line can be run using the Docker
container built using instructions in the top-level README).

Some examples of command lines and configurations for `depth=2` and `depth=4`
are in the `tests/` directory.

### Pre-training to Accuracy

BERT models can be pre-trained using a command like,

```bash
python -u -m launch --nnodes 2 --node_rank 0 --nproc_per_node=8 main_with_runtime.py \
        --master_addr 10.185.12.207 \
        --module models.bert24.depth=8 \
        --config_path models/bert24/depth=8/dp_conf.json \
        --max_seq_length 128 \
        --train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --train_path <TRAIN_PATH> \
        --partitioned_data \
        --bert_config_path configs/bert_config_bert-large-uncased.json \
        --vocab_path <VOCAB_PATH> \
        --do_train \
        --on_memory \
        --do_lower_case \
        --num_train_epochs 30.0 \
        --learning_rate 2e-5 \
        --checkpoint_dir <CHECKPOINT_DIR>
```

## Fine-tuning Pre-trained Model

A partitioned pre-trained BERT model can be fine-tuned for various downstream
GLUE tasks using a command like,

```bash
python -u -m torch.distributed.launch --nproc_per_node=8 compute_glue_scores.py \
    --data_dir <DATA_DIR>  --task_name MNLI --do_train --do_eval \
    -m models.bert24.depth=8 \
    --bert_config_path configs/bert_config_bert-large-uncased.json \
    --checkpoint_path <CHECKPOINT_PATH> \
    --epoch 5 --num_stages 1
```

The `--num_stages` argument specifies how many stages the pre-training step
was divided into, and the `--epoch` argument specifies the epoch of the
checkpoint to use.
