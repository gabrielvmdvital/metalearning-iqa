#!/bin/bash

#SBATCH --job-name=metalearning_iqa
#SBATCH -p short-complex
#SBATCH --nodelist=cluster-node9
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH --output=job_output_metalearning.txt
#SBATCH --error=job_error_metalearning.txt

cd $HOME/metalearning/projeto

source .venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1

echo "Start time: $(date)"

echo "==================== INICIANDO PIPELINE DE META-LEARNING IQA ===================="

python main.py

echo "==================== PIPELINE CONCLUÍDO ===================="

echo "Finish time: $(date)"
