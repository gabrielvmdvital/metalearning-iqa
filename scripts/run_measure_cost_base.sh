#!/bin/bash

#SBATCH --job-name=measure_cost_base_iqa
#SBATCH -p short-complex
#SBATCH --nodelist=cluster-node6
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --output=job_output_measure_cost_base.txt
#SBATCH --error=job_error_measure_cost_base.txt

cd $HOME/metalearning/metalearning-iqa

# Ativando o ambiente virtual Linux do cluster
source .venv/bin/activate

echo "Start time: $(date)"

echo "==================== MEDINDO CUSTO DA MATRIZ BASE (S/ VISÃO COMPUTACIONAL) ===================="

python measure_cost_base.py

echo "==================== MEDIÇÃO CONCLUÍDA ===================="

echo "Finish time: $(date)"
