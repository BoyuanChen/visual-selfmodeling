
#!/bin/bash

screen -S train -dm bash -c "CUDA_VISIBLE_DEVICES=1 python ../eval.py ../configs/single_reconstruct/config1.yaml ./logs_reconstruction_1/lightning_logs/version_0/checkpoints eval-single; \
                             exec sh";