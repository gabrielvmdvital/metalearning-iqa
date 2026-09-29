#!/bin/bash

#SBATCH --job-name=measure_cost_iqa
#SBATCH -p short-complex
#SBATCH --nodelist=cluster-node6
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --output=job_output_measure_cost.txt
#SBATCH --error=job_error_measure_cost.txt

cd $HOME/metalearning/metalearning-iqa

# Ativando o ambiente virtual Linux do cluster
source .venv/bin/activate

echo "Start time: $(date)"

echo "==================== MEDINDO CUSTO COMPUTACIONAL DAS FEATURES ===================="

python measure_cost.py

echo "==================== MEDIÇÃO CONCLUÍDA ===================="

echo "Finish time: $(date)"
