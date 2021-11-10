#!/bin/bash

screen -S train -dm bash -c "CUDA_VISIBLE_DEVICES=0 python ../main.py ../configs/single_reconstruct/config1.yaml; \
                             exec sh";