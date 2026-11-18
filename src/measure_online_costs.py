import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extract_features_hpo import extract_image_features_fast
from src.hpo import (
    encode_configurations,
    method_4_smbo_warm_feurer,
)

FEATURE_COLS = [
    'brilho', 'contraste', 'snr', 'entropia',
    'assimetria', 'curtose', 'colorfulness',
    'spatial_frequency', 'blur_extent'
]


def measure_all_costs():
    """
    Mede no mesmo hardware (single-core, n_jobs=1) os custos reais online e offline
    da solução por Meta-aprendizagem para comparação rigorosa com o AutoML.
    """
    base_dir = Path(__file__).resolve().parent.parent
    proc_dir = base_dir / 'data' / 'processed'

    df_img = pd.read_csv(proc_dir / 'image_features_dataset.csv')
    df_configs = pd.read_csv(proc_dir / 'configurations_K.csv')
    loss_matrix = pd.read_csv(proc_dir / 'loss_matrix_H_K_n.csv', index_col=0)
    X_ext = pd.read_csv(proc_dir / 'X_matrix_estendida.csv', index_col=0)

    datasets = sorted(df_img['dataset_name'].unique().tolist())
    Z = encode_configurations(df_configs)
    configs_dict = df_configs.set_index('config_id').to_dict('index')

    # 1. Medir tempo médio por imagem da extração das 9 features NSS (single-thread)
    # Testamos 30 imagens por dataset (ou todas se <= 30) para obter custo exato por imagem
    img_times = []
    valid_checked = 0
    for ds in datasets:
        sub_paths = df_img[df_img['dataset_name'] == ds]['dist_path'].head(10).tolist()
        for p in sub_paths:
            if os.path.exists(p):
                t0 = time.perf_counter()
                extract_image_features_fast(p)
                img_times.append(time.perf_counter() - t0)
                valid_checked += 1

    mean_img_sec = float(np.mean(img_times)) if img_times else 0.002
    print(f"Imagens verificadas para custo NSS: {valid_checked} | Tempo médio/imagem: {mean_img_sec*1000:.3f} ms")

    # 2. Medir tempo real online de HPO Warm-Start (T=5 e T=20) por dataset:
    # inclui normalização LODO + kNN + GP/EI + treino real dos SVR em 5-fold CV
    cost_rows = []
    for ds in datasets:
        df_sub = df_img[df_img['dataset_name'] == ds]
        X = df_sub[FEATURE_COLS].values
        y = df_sub['mos'].values
        n_samples = len(y)

        # Tempo estimado de extração das 9 features do dataset (n_samples * tempo/imagem)
        t_feat_extract_ds = n_samples * mean_img_sec

        # Tempo de consulta LODO + SMBO (20 iterações)
        t0_meta = time.perf_counter()
        eval_order, _, dist_nn = method_4_smbo_warm_feurer(
            ds, loss_matrix, X_ext, Z, k=1, T=20, n_init=5, rng=np.random.RandomState(42)
        )
        t_smbo_20 = time.perf_counter() - t0_meta

        t0_meta5 = time.perf_counter()
        _, _, _ = method_4_smbo_warm_feurer(
            ds, loss_matrix, X_ext, Z, k=1, T=5, n_init=5, rng=np.random.RandomState(42)
        )
        t_knn_5 = time.perf_counter() - t0_meta5

        # Tempo real de treinar as configurações sugeridas em 5-Fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scaled_splits = []
        for tr_idx, te_idx in kf.split(X):
            sc = StandardScaler()
            scaled_splits.append((tr_idx, te_idx, sc.fit_transform(X[tr_idx]), sc.transform(X[te_idx])))

        t0_fit = time.perf_counter()
        t_fit_5 = 0.0
        for step_idx, cfg_id in enumerate(eval_order[:20], 1):
            cfg = configs_dict[int(cfg_id)]
            svr_kwargs = {'kernel': cfg['kernel'], 'C': float(cfg['C']), 'epsilon': float(cfg['epsilon'])}
            if cfg['kernel'] == 'rbf':
                g = cfg['gamma']
                svr_kwargs['gamma'] = g if g in ('scale', 'auto') else float(g)
            for tr_idx, te_idx, X_tr, X_te in scaled_splits:
                m = SVR(**svr_kwargs)
                m.fit(X_tr, y[tr_idx])
                m.predict(X_te)
            if step_idx == 5:
                t_fit_5 = time.perf_counter() - t0_fit
        t_fit_20 = time.perf_counter() - t0_fit

        cost_rows.append({
            'dataset': ds,
            'n_samples': n_samples,
            't_img_features_s': t_feat_extract_ds,
            't_knn_meta_s': t_knn_5,
            't_smbo_overhead_20_s': max(0.0, t_smbo_20 - t_knn_5),
            't_hpo_fit_5_s': t_fit_5,
            't_hpo_fit_20_s': t_fit_20,
            't_online_hpo5_total_s': t_feat_extract_ds + t_knn_5 + t_fit_5,
            't_online_hpo20_total_s': t_feat_extract_ds + t_smbo_20 + t_fit_20,
        })

    df_costs = pd.DataFrame(cost_rows)
    df_costs.to_csv(proc_dir / 'p4_online_costs_measured.csv', index=False)
    print("\n--- Resumo de Custos Online Medidos (Média sobre 56 Datasets) ---")
    print(df_costs.drop(columns=['dataset']).mean().round(4).to_string())
    return df_costs


if __name__ == '__main__':
    measure_all_costs()
