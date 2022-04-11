#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m exp.coo.annotate --batch_size 64 --idxs 0,24 --range > ./logs/annotate_gpu0.log 2>&1&
CUDA_VISIBLE_DEVICES=1 python -m exp.coo.annotate --batch_size 64 --idxs 25,50 --range > ./logs/annotate_gpu1.log 2>&1&
CUDA_VISIBLE_DEVICES=2 python -m exp.coo.annotate --batch_size 64 --idxs 51,75 --range > ./logs/annotate_gpu2.log 2>&1&
CUDA_VISIBLE_DEVICES=3 python -m exp.coo.annotate --batch_size 64 --idxs 76,100 --range > ./logs/annotate_gpu3.log 2>&1&
