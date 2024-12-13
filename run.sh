#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 main.py --cfg ./cfgs/mlla_t.yaml --data-path /raid/dzz/art/art_dataset --output /raid/dzz/art/output --amp