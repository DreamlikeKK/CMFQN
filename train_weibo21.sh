#!/bin/bash

python ./Main.py -train_dataset weibo21 \
                -test_dataset weibo21 \
                -batch_size 16 \
                -epochs 100 \
                -val 0 \
                -mask 0 \
                -que 0.6
