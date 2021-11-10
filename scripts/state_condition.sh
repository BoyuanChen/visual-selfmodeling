#!/bin/bash

screen -S train -dm bash -c "CUDA_VISIBLE_DEVICES=1 python ../main.py ../configs/state_condition/config1.yaml NA NA; \
                             exec sh";