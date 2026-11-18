import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

FEATURE_COLS = [
    'brilho', 'contraste', 'snr', 'entropia',
    'assimetria', 'curtose', 'colorfulness',
    'spatial_frequency', 'blur_extent'
]


def get_tabular_algorithms(random_state=42):
    """
    Retorna o portfólio de m=6 algoritmos regressores tabulares com suas
    configurações padrão, restritos a n_jobs=1 para comparabilidade justa de CPU.
    """
    return {
        'SVR': lambda: SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1),
        'ExtraTrees': lambda: ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=1),
        'LightGBM': lambda: LGBMRegressor(random_state=random_state, verbose=-1, n_jobs=1),
        'kNN': lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=1),
        'MLP': lambda: MLPRegressor(max_iter=300, random_state=random_state),
    }


def _safe_spearman(y_true, y_pred):
    """Calcula SRCC tratando predições constantes ou NaN."""
    if len(y_true) < 2 or np.all(y_pred == y_pred[0]) or np.isnan(y_pred).any():
        return 0.0
    val = spearmanr(y_true, y_pred).statistic
    return float(val) if not np.isnan(val) else 0.0


def build_tabular_performance_matrices():
    """
    Avalia os m=6 algoritmos tabulares em configuração padrão sobre os 56 datasets
    de IQA usando exatamente o mesmo protocolo 5-Fold CV da Prática 3.
    Gera P_matrix_tabular, L_matrix_tabular, R_matrix_tabular, folds e tempos.
    """
    base_dir = Path(__file__).resolve().parent.parent
    proc_dir = base_dir / 'data' / 'processed'
    features_file = proc_dir / 'image_features_dataset.csv'

    if not features_file.exists():
        raise FileNotFoundError(f"Arquivo {features_file} não encontrado.")

    df_all = pd.read_csv(features_file)
    dataset_names = sorted(df_all['dataset_name'].unique().tolist())
    alg_factories = get_tabular_algorithms(random_state=42)
    alg_names = list(alg_factories.keys())

    p_rows = []
    l_rows = []
    time_rows = []
    fold_rows = []

    print(f"Construindo Matriz P_tabular ({len(dataset_names)} datasets x {len(alg_names)} algoritmos)...")
    t_global = time.perf_counter()

    for ds in dataset_names:
        df_sub = df_all[df_all['dataset_name'] == ds]
        X = df_sub[FEATURE_COLS].values
        y = df_sub['mos'].values
        n_samples = len(y)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(X))

        # Pré-escalonar cada fold estritamente sem vazamento (fit só no treino)
        scaled_splits = []
        for tr_idx, te_idx in splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])
            scaled_splits.append((tr_idx, te_idx, X_tr, X_te))

        p_row = {'dataset': ds}
        l_row = {'dataset': ds}
        t_row = {'dataset': ds}
        per_fold_scores = {f_i: {'dataset': ds, 'fold': f_i} for f_i in range(len(splits))}

        for alg_name in alg_names:
            factory = alg_factories[alg_name]
            y_pred_oof = np.zeros(n_samples)

            t0 = time.perf_counter()
            for f_i, (tr_idx, te_idx, X_tr, X_te) in enumerate(scaled_splits):
                model = factory()
                model.fit(X_tr, y[tr_idx])
                preds_fold = model.predict(X_te)
                y_pred_oof[te_idx] = preds_fold
                per_fold_scores[f_i][alg_name] = _safe_spearman(y[te_idx], preds_fold)
            elapsed = time.perf_counter() - t0

            srcc_oof = _safe_spearman(y, y_pred_oof)
            p_row[alg_name] = srcc_oof
            l_row[alg_name] = 1.0 - srcc_oof
            t_row[alg_name] = elapsed

        p_rows.append(p_row)
        l_rows.append(l_row)
        time_rows.append(t_row)
        for f_i in range(len(splits)):
            fold_rows.append(per_fold_scores[f_i])

    df_p = pd.DataFrame(p_rows).set_index('dataset')
    df_l = pd.DataFrame(l_rows).set_index('dataset')
    df_t = pd.DataFrame(time_rows).set_index('dataset')
    df_folds = pd.DataFrame(fold_rows)

    # Matriz de Ranks R (1 = melhor / maior SRCC, empates com rank médio)
    df_r = df_p.apply(lambda row: rankdata(-row, method='average'), axis=1, result_type='broadcast')

    df_p.to_csv(proc_dir / 'P_matrix_tabular.csv')
    df_l.to_csv(proc_dir / 'L_matrix_tabular.csv')
    df_r.to_csv(proc_dir / 'R_matrix_tabular.csv')
    df_t.to_csv(proc_dir / 'P_matrix_tabular_times.csv')
    df_folds.to_csv(proc_dir / 'P_matrix_tabular_folds.csv', index=False)

    total_time = time.perf_counter() - t_global
    print(f"Matrizes P_tabular geradas em {total_time:.2f}s!")
    print("\n--- Desempenho Médio (Perda L = 1 - SRCC) e Tempo Médio por Dataset ---")
    summary = pd.DataFrame({
        'SRCC Médio': df_p.mean(),
        'Perda L Média': df_l.mean(),
        'Rank Médio': df_r.mean(),
        'Tempo Médio/DS (s)': df_t.mean(),
        'Tempo Total 56 DS (s)': df_t.sum(),
    }).sort_values('Rank Médio')
    print(summary.round(4).to_string())

    # Verificação de consistência com a Config 0 (SVR default) da Prática 3
    loss_p3_path = proc_dir / 'loss_matrix_H_K_n.csv'
    if loss_p3_path.exists():
        loss_p3 = pd.read_csv(loss_p3_path, index_col=0)
        svr_p3 = loss_p3.iloc[0].reindex(df_l.index)
        max_diff = np.max(np.abs(df_l['SVR'] - svr_p3))
        print(f"\n[Verificação Protocolo P3] Diferença máxima SVR default vs Config 0 (Prática 3): {max_diff:.6e}")

    return df_p, df_l, df_r, df_t, df_folds


if __name__ == '__main__':
    build_tabular_performance_matrices()
