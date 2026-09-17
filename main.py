# main.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from src.data_loader import build_master_dataframe
from src.evaluation import get_metrics_dict, calculate_performance_matrix
from src.features import build_meta_features_matrix
from src.models import BaselineRanker, SignificantWinsRanker, MetaRegressor, HarrisForest
from src.experiments import leave_one_dataset_out_evaluation, plot_critical_difference_diagram

def main():
    print("=== Iniciando o Pipeline de Meta-Learning para IQA ===")

    base_dir = Path(__file__).resolve().parent
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
    P_matrix_path = data_processed_dir / 'P_matrix_folds.csv'
    if not P_matrix_path.exists():
        metrics = get_metrics_dict()
        df_P_folds = calculate_performance_matrix(df_metadata, metrics)
        df_P_folds.to_csv(P_matrix_path, index=False)
    else:
        df_P_folds = pd.read_csv(P_matrix_path)
        metrics = {c: None for c in df_P_folds.columns if c not in ['dataset', 'fold']}
    
    P_matrix = df_P_folds.groupby('dataset')[list(metrics.keys())].mean()
    
    print("=== Passo 2: Construindo Matriz X (Base e Estendida) ===")
    df_X_base = build_meta_features_matrix(df_metadata, extended=False)
    df_X_base = df_X_base.sort_values('dataset').reset_index(drop=True)
    df_X_base.to_csv(data_processed_dir / 'X_matrix_base.csv', index=False)
    X_matrix_base = df_X_base.drop('dataset', axis=1)

    df_X_estendida = build_meta_features_matrix(df_metadata, extended=True)
    df_X_estendida = df_X_estendida.sort_values('dataset').reset_index(drop=True)
    df_X_estendida.to_csv(data_processed_dir / 'X_matrix_estendida.csv', index=False)
    X_matrix_estendida = df_X_estendida.drop('dataset', axis=1)
    
    print("=== Passo 3: Construindo Matriz R ===")
    R_matrix = P_matrix.apply(lambda row: rankdata(-row, method='average'), axis=1, result_type='broadcast')
    R_matrix.to_csv(data_processed_dir / 'R_matrix.csv', index=True)
    
    print("\n=== Passo 4: Inicializando Abordagens ===")
    models = {
        'AR': BaselineRanker(method='AR'),
        'MR': BaselineRanker(method='MR'),
        'Abordagem1 (P)': MetaRegressor(target='P'),
        'Abordagem2 (R)': MetaRegressor(target='R'),
        'HARRIS (L=0.5)': HarrisForest(lambd=0.5),
    }
    
    print("\n=== Passo 5: Avaliação Leave-One-Dataset-Out (BASE) ===")
    results_base = leave_one_dataset_out_evaluation(X_matrix_base, P_matrix, R_matrix, models)

    print("\n=== Passo 5: Avaliação Leave-One-Dataset-Out (ESTENDIDA) ===")
    results_estendida = leave_one_dataset_out_evaluation(X_matrix_estendida, P_matrix, R_matrix, models)
    
    print("\n=== Passo 6: Gerando Gráficos Comparativos ===")
    
    # Loss curves comparison
    plt.figure(figsize=(12, 8))
    for name, res in results_base.items():
        if name not in ['AR', 'MR']:
            plt.plot(range(1, len(res['Mean_Curve'])+1), res['Mean_Curve'], linestyle='--', label=f"{name} BASE (AUC={res['Mean_AUC']:.3f})")
    
    for name, res in results_estendida.items():
        plt.plot(range(1, len(res['Mean_Curve'])+1), res['Mean_Curve'], label=f"{name} EST (AUC={res['Mean_AUC']:.3f})")
        
    plt.xlabel('Número de testes (t)')
    plt.ylabel('Perda Média')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Curvas de Perda das Abordagens - Base vs Estendida')
    plt.tight_layout()
    plt.savefig(notebooks_dir / 'loss_curves_comparison.png')
    plt.close()
    
    # CD Diagram
    dict_srcc = {}
    for name, res in results_base.items():
        if name in ['AR', 'MR']:
            dict_srcc[name] = res['All_SRCC']
        else:
            dict_srcc[name + " (Base)"] = res['All_SRCC']
            
    for name, res in results_estendida.items():
        if name not in ['AR', 'MR']:
            dict_srcc[name + " (Estendida)"] = res['All_SRCC']

    plot_critical_difference_diagram(dict_srcc, alpha=0.05, save_path=notebooks_dir / 'cd_diagram_comparison.png')

if __name__ == '__main__':
    main()
