#!/bin/bash
#SBATCH --job-name=gpt120b_research
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G               # 120B needs more system RAM to "unpack"
#SBATCH --time=03:00:00          # Set a longer window for your research
#SBATCH --output=vllm_120b_%j.log

# NOTE: INSTALL VLLM TO UR OWN CONDA SETUP
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpt_oss

# We use --async-scheduling for the 120B MoE architecture
vllm serve openai/gpt-oss-120b \                                                                                           
    --quantization mxfp4 \                                                                                                 
    --max-model-len 128000 \                                                                                               
    --gpu-memory-utilization 0.95 \                                                                                        
    --async-scheduling \                                                                                                   
    --trust-remote-code \                                                                                                  
    --port 8050 