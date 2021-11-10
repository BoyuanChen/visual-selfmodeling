
#!/bin/bash

screen -S train -dm bash -c "CUDA_VISIBLE_DEVICES=3 python ../eval.py ../configs/state_condition/config1.yaml ./logs_state-condition_1/lightning_logs/version_0/checkpoints eval-state-condition; \
                             exec sh";