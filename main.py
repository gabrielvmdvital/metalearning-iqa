# src/main_pipeline.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from data_loader import build_master_dataframe
from evaluation import get_metrics_dict, calculate_performance_matrix
from features import build_meta_features_matrix
from models import BaselineRanker, SignificantWinsRanker, MetaRegressor, HarrisForest
from experiments import leave_one_dataset_out_evaluation

def main():
    print("=== Iniciando o Pipeline de Meta-Learning para IQA ===")

    base_dir = Path(__file__).resolve().parent.parent
    data_raw_dir = base_dir / 'data' / 'raw'
    data_processed_dir = base_dir / 'data' / 'processed'
    notebooks_dir = base_dir / 'notebooks'
    
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    df_metadata = build_master_dataframe(data_raw_dir)
    
    if df_metadata.empty:
        print("Finalizando execução pois não há dados carregados.")
        return
        
    print("\n=== Passo 1: Construindo Matriz P ===")
    metrics = get_metrics_dict()
    df_P_folds = calculate_performance_matrix(df_metadata, metrics)
    df_P_folds.to_csv(data_processed_dir / 'P_matrix_folds.csv', index=False)
    
    P_matrix = df_P_folds.groupby('dataset')[list(metrics.keys())].mean()
    
    print("=== Passo 2: Construindo Matriz X ===")
    df_X = build_meta_features_matrix(df_metadata)
    df_X = df_X.sort_values('dataset').reset_index(drop=True)
    df_X.to_csv(data_processed_dir / 'X_matrix.csv', index=False)
    X_matrix = df_X.drop('dataset', axis=1)
    
    print("=== Passo 3: Construindo Matriz R ===")
    R_matrix = P_matrix.apply(lambda row: rankdata(-row, method='average'), axis=1, result_type='broadcast')
    R_matrix.to_csv(data_processed_dir / 'R_matrix.csv', index=True)
    
    print("\n=== Passo 4: Inicializando Abordagens ===")
    models = {
        'AR': BaselineRanker(method='AR'),
        'MR': BaselineRanker(method='MR'),
        # 'SigWins': SignificantWinsRanker(), # Requer fit customizado passando P_folds
        'Abordagem1 (P)': MetaRegressor(target='P'),
        'Abordagem2 (R)': MetaRegressor(target='R'),
        'HARRIS (L=0.0)': HarrisForest(lambd=0.0),
        'HARRIS (L=0.5)': HarrisForest(lambd=0.5),
        'HARRIS (L=1.0)': HarrisForest(lambd=1.0),
    }
    
    print("\n=== Passo 5: Avaliação Leave-One-Dataset-Out ===")
    results = leave_one_dataset_out_evaluation(X_matrix, P_matrix, R_matrix, models)
    
    print("\n=== Passo 6: Gerando Gráficos ===")
    
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.plot(range(1, len(res['Mean_Curve'])+1), res['Mean_Curve'], label=f"{name} (AUC={res['Mean_AUC']:.3f})")
    plt.xlabel('Número de testes (t)')
    plt.ylabel('Perda Média')
    plt.legend()
    plt.title('Curvas de Perda das Abordagens')
    plt.savefig(notebooks_dir / 'loss_curves.png')
    plt.close()
    
    from experiments import plot_critical_difference_diagram
    dict_srcc = {name: res['All_SRCC'] for name, res in results.items()}
    plot_critical_difference_diagram(dict_srcc, alpha=0.05, save_path=notebooks_dir / 'cd_diagram.png')

if __name__ == '__main__':
    main()
