#!/bin/bash

python3 main.py --inference --model FlowNet2 --save_flow --inference_dataset MpiSintelClean \
                --inference_dataset_root dataset/MPI-Sintel-training_images/training \
                --resume /model/FlowNet2_checkpoint.pth.tar