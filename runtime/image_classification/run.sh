python main_with_runtime.py --module models.$1.gpus=4 -b 64 --data_dir /data/ImageNet --rank 3 --local_rank 3 --master_addr localhost --config_path models/$1/gpus=4/hybrid_conf.json --distributed_backend gloo --num_ranks_in_server 4 --checkpoint_dir /data/models/$1 &
python main_with_runtime.py --module models.$1.gpus=4 -b 64 --data_dir /data/ImageNet --rank 1 --local_rank 1 --master_addr localhost --config_path models/$1/gpus=4/hybrid_conf.json --distributed_backend gloo --num_ranks_in_server 4 --checkpoint_dir /data/models/$1 &
python main_with_runtime.py --module models.$1.gpus=4 -b 64 --data_dir /data/ImageNet --rank 2 --local_rank 2 --master_addr localhost --config_path models/$1/gpus=4/hybrid_conf.json --distributed_backend gloo --num_ranks_in_server 4 --checkpoint_dir /data/models/$1 &
python main_with_runtime.py --module models.$1.gpus=4 -b 64 --data_dir /data/ImageNet --rank 0 --local_rank 0 --master_addr localhost --config_path models/$1/gpus=4/hybrid_conf.json --distributed_backend gloo --num_ranks_in_server 4 --checkpoint_dir /data/models/$1