import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

FEATURE_COLS = [
    'brilho', 'contraste', 'snr', 'entropia', 
    'assimetria', 'curtose', 'colorfulness', 
    'spatial_frequency', 'blur_extent'
]

def generate_configurations(K=75, seed=42):
    """
    Gera o conjunto fixo de K configurações no espaço Theta.
    Configuração 0: Padrão do scikit-learn (C=1.0, kernel='rbf', gamma='scale', epsilon=0.1).
    Configurações 1 a K-1: Amostragem quase-aleatória estratificada em escala logarítmica.
    """
    rng = np.random.RandomState(seed)
    configs = []
    
    # 0: Configuração padrão da biblioteca scikit-learn
    configs.append({
        'config_id': 0,
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'epsilon': 0.1,
        'is_default': True
    })
    
    # Restantes K-1 configurações
    # ~75% RBF, ~25% Linear
    n_remaining = K - 1
    n_rbf = int(n_remaining * 0.75)
    n_linear = n_remaining - n_rbf
    
    # RBF configs
    for i in range(n_rbf):
        c_val = float(10 ** rng.uniform(-2.0, 3.0)) # 0.01 a 1000.0
        eps_val = float(10 ** rng.uniform(-3.0, np.log10(0.5))) # 0.001 a 0.5
        gamma_choice = rng.choice(['scale', 'auto', 'numeric'], p=[0.25, 0.15, 0.60])
        if gamma_choice == 'numeric':
            gamma_val = float(10 ** rng.uniform(-4.0, 1.0)) # 0.0001 a 10.0
        else:
            gamma_val = gamma_choice
            
        configs.append({
            'config_id': len(configs),
            'kernel': 'rbf',
            'C': round(c_val, 5),
            'gamma': gamma_val if isinstance(gamma_val, str) else round(gamma_val, 6),
            'epsilon': round(eps_val, 5),
            'is_default': False
        })
        
    # Linear configs
    for i in range(n_linear):
        c_val = float(10 ** rng.uniform(-2.0, 3.0))
        eps_val = float(10 ** rng.uniform(-3.0, np.log10(0.5)))
        configs.append({
            'config_id': len(configs),
            'kernel': 'linear',
            'C': round(c_val, 5),
            'gamma': 'scale', # sem efeito em linear, mantido para consistência de schema
            'epsilon': round(eps_val, 5),
            'is_default': False
        })
        
    df_configs = pd.DataFrame(configs)
    return df_configs

def evaluate_dataset_configs(args):
    """
    Avalia todas as K configurações para um dataset via 5-Fold Cross Validation.
    Retorna uma tupla (dataset_name, losses_list, srcc_list).
    """
    dataset_name, df_sub, configs = args
    X = df_sub[FEATURE_COLS].values
    y = df_sub['mos'].values
    n_samples = len(y)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(X))
    
    losses = []
    srccs = []
    
    for cfg in configs:
        svr_kwargs = {
            'kernel': cfg['kernel'],
            'C': cfg['C'],
            'epsilon': cfg['epsilon']
        }
        if cfg['kernel'] == 'rbf':
            svr_kwargs['gamma'] = cfg['gamma']
            
        y_pred = np.zeros(n_samples)
        
        for train_idx, val_idx in splits:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va = X[val_idx]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            
            model = SVR(**svr_kwargs)
            model.fit(X_tr_sc, y_tr)
            y_pred[val_idx] = model.predict(X_va_sc)
            
        # Calcular SRCC (Spearman Rank Correlation)
        if np.all(y_pred == y_pred[0]) or np.isnan(y_pred).any():
            srcc = 0.0
        else:
            srcc_val = spearmanr(y, y_pred).statistic
            srcc = float(srcc_val) if not np.isnan(srcc_val) else 0.0
            
        loss = 1.0 - srcc
        losses.append(loss)
        srccs.append(srcc)
        
    return dataset_name, losses, srccs

def build_loss_matrix():
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / 'data' / 'processed'
    features_file = processed_dir / 'image_features_dataset.csv'
    
    if not features_file.exists():
        raise FileNotFoundError(f"Arquivo de features {features_file} não encontrado. Execute extract_features_hpo.py primeiro.")
        
    print(f"Carregando features de imagens de {features_file}...")
    df_all = pd.read_csv(features_file)
    dataset_names = sorted(df_all['dataset_name'].unique().tolist())
    n_datasets = len(dataset_names)
    print(f"Encontrados {n_datasets} datasets.")
    
    K = 75
    print(f"Gerando K={K} configurações de hiperparâmetros (Config 0 = padrão)...")
    df_configs = generate_configurations(K=K, seed=42)
    configs_file = processed_dir / 'configurations_K.csv'
    df_configs.to_csv(configs_file, index=False)
    print(f"Configurações salvas em: {configs_file}")
    
    configs_list = df_configs.to_dict('records')
    
    tasks = []
    for d_name in dataset_names:
        df_sub = df_all[df_all['dataset_name'] == d_name].copy()
        tasks.append((d_name, df_sub, configs_list))
        
    print(f"Avaliando {K} configurações em {n_datasets} datasets com 5-fold CV...")
    t0 = time.time()
    
    # Processar em paralelo com ThreadPoolExecutor
    loss_dict = {}
    srcc_dict = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(evaluate_dataset_configs, tasks))
        
    for d_name, losses, srccs in results:
        loss_dict[d_name] = losses
        srcc_dict[d_name] = srccs
        
    # Matriz K x n (linhas = configurações, colunas = datasets)
    loss_matrix_K_n = pd.DataFrame(loss_dict)
    loss_matrix_K_n.index.name = 'config_id'
    
    output_loss_Kn = processed_dir / 'loss_matrix_H_K_n.csv'
    loss_matrix_K_n.to_csv(output_loss_Kn)
    
    # Matriz n x K (linhas = datasets, colunas = configurações)
    loss_matrix_n_K = loss_matrix_K_n.T
    loss_matrix_n_K.columns = [f"config_{i}" for i in range(K)]
    loss_matrix_n_K.index.name = 'dataset_name'
    output_loss_nK = processed_dir / 'loss_matrix_datasets_x_configs.csv'
    loss_matrix_n_K.to_csv(output_loss_nK)
    
    # Salvar também matriz de SRCC correspondente
    srcc_matrix_K_n = pd.DataFrame(srcc_dict)
    srcc_matrix_K_n.index.name = 'config_id'
    output_srcc = processed_dir / 'srcc_matrix_H_K_n.csv'
    srcc_matrix_K_n.to_csv(output_srcc)
    
    t1 = time.time()
    print(f"\nMatriz de perda K x n ({K} x {n_datasets}) gerada com sucesso em {t1 - t0:.2f} segundos!")
    print(f"Salva em: {output_loss_Kn}")
    print(f"Matriz transposta salva em: {output_loss_nK}")
    print(f"Matriz de SRCC salva em: {output_srcc}")
    
    # Estatísticas sumárias
    print("\nResumo da Matriz de Perda (L = 1 - SRCC):")
    min_losses = loss_matrix_K_n.min(axis=0) # L* ótimo por dataset
    max_losses = loss_matrix_K_n.max(axis=0) # Pior perda por dataset
    default_losses = loss_matrix_K_n.iloc[0] # Perda da configuração padrão
    
    summary = pd.DataFrame({
        'L_star (ótimo)': min_losses,
        'L_default (padrão)': default_losses,
        'L_pior (pior)': max_losses,
        'Gap (default - ótimo)': default_losses - min_losses
    })
    print(summary.describe().round(4))
    
    return loss_matrix_K_n

if __name__ == '__main__':
    build_loss_matrix()
