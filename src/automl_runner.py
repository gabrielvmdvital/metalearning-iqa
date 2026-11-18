import json
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

FEATURE_COLS = [
    'brilho', 'contraste', 'snr', 'entropia',
    'assimetria', 'curtose', 'colorfulness',
    'spatial_frequency', 'blur_extent'
]


def srcc_loss_metric(
    X_val, y_val, estimator, labels,
    X_train, y_train, weight_val=None, weight_train=None,
    config=None, groups_val=None, groups_train=None,
):
    """Função de perda L = 1 - SRCC para otimização interna do FLAML."""
    y_pred = estimator.predict(X_val)
    if len(y_val) < 2 or np.all(y_pred == y_pred[0]) or np.isnan(y_pred).any():
        return 1.0, {'srcc_loss': 1.0}
    r = spearmanr(y_val, y_pred).statistic
    loss = 1.0 - (float(r) if not np.isnan(r) else 0.0)
    return loss, {'srcc_loss': loss}


def _safe_spearman(y_true, y_pred):
    """Calcula SRCC tratando predições constantes ou NaN."""
    if len(y_true) < 2 or np.all(y_pred == y_pred[0]) or np.isnan(y_pred).any():
        return 0.0
    r = spearmanr(y_true, y_pred).statistic
    return float(r) if not np.isnan(r) else 0.0


def _sanitize_for_json(obj):
    """Converte tipos NumPy para tipos nativos Python serializáveis em JSON."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _run_flaml_on_dataset(args):
    """
    Executa o FLAML em um único dataset sob o mesmo protocolo 5-Fold CV da Prática 3.
    O FLAML recebe estritamente os dados de treino do fold (sem tocar no teste).
    """
    dataset_name, df_sub, mode_name, time_budget_fold, starting_points, ensemble, seed = args

    X = df_sub[FEATURE_COLS].values
    y = df_sub['mos'].values
    n_samples = len(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(X))

    y_pred_oof = np.zeros(n_samples)
    fold_srccs = []
    best_estimators = []
    best_configs = []
    eval_methods = []
    ensemble_members = []
    internal_val_losses = []
    total_trials = 0
    learner_trials = Counter()

    t0_total = time.perf_counter()

    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        automl = AutoML()
        try:
            automl.fit(
                X_tr,
                y_tr,
                task='regression',
                metric=srcc_loss_metric,
                time_budget=time_budget_fold,
                starting_points=starting_points,
                ensemble=ensemble,
                eval_method='auto',
                n_jobs=1,
                verbose=0,
                seed=seed + fold_idx,
            )
            preds = automl.predict(X_te)
        except Exception as e:
            # Fallback seguro se ocorrer erro numérico em algum fold extremo
            preds = np.full(len(te_idx), np.mean(y_tr))
            best_estimators.append(f'fallback_error:{type(e).__name__}')
            best_configs.append({})
            eval_methods.append('error')
            internal_val_losses.append(1.0)
            y_pred_oof[te_idx] = preds
            fold_srccs.append(0.0)
            continue

        y_pred_oof[te_idx] = preds
        fold_srccs.append(_safe_spearman(y_te, preds))

        best_est = str(automl.best_estimator)
        best_estimators.append(best_est)
        best_configs.append(_sanitize_for_json(automl.best_config or {}))
        eval_methods.append(str(getattr(automl._state, 'eval_method', 'auto')))
        internal_val_losses.append(float(automl.best_loss) if automl.best_loss is not None else 1.0)

        iter_map = getattr(automl, '_iter_per_learner', {}) or {}
        for k, v in iter_map.items():
            learner_trials[str(k)] += int(v)
            total_trials += int(v)

        if ensemble and hasattr(automl.model, 'named_estimators_'):
            ensemble_members.append(list(automl.model.named_estimators_.keys()))
        else:
            ensemble_members.append([best_est])

    wall_time_sec = time.perf_counter() - t0_total
    srcc_oof = _safe_spearman(y, y_pred_oof)
    srcc_folds_mean = float(np.mean(fold_srccs))
    modal_est = Counter(best_estimators).most_common(1)[0][0] if best_estimators else 'none'

    return {
        'dataset': dataset_name,
        'mode': mode_name,
        'n_samples': n_samples,
        'time_budget_ds': time_budget_fold * 5.0,
        'starting_points': str(starting_points),
        'ensemble': bool(ensemble),
        'srcc_oof': srcc_oof,
        'loss_oof': 1.0 - srcc_oof,
        'srcc_folds_mean': srcc_folds_mean,
        'loss_folds_mean': 1.0 - srcc_folds_mean,
        'wall_time_sec': wall_time_sec,
        'best_estimator_modal': modal_est,
        'best_estimators_folds': json.dumps(best_estimators),
        'best_configs_folds': json.dumps(best_configs),
        'eval_methods_folds': json.dumps(eval_methods),
        'n_trials_total': int(total_trials),
        'trials_per_learner': json.dumps(dict(learner_trials)),
        'ensemble_members_folds': json.dumps(ensemble_members),
        'internal_val_loss_mean': float(np.mean(internal_val_losses)),
    }


def get_automl_modes():
    """
    Define as execuções do FLAML:
    - 3 orçamentos principais (com meta-conhecimento OpenML nativo starting_points='data'):
      curto (1.5s/ds), médio (5.0s/ds), longo (15.0s/ds).
    - Ablação sem meta-aprendizagem (starting_points='static') em curto e médio.
    - Ablação com ensemble stacking (ensemble=True) em médio.
    """
    return [
        ('flaml_curto', 0.3, 'data', False, 100),
        ('flaml_medio', 1.0, 'data', False, 200),
        ('flaml_longo', 3.0, 'data', False, 300),
        ('flaml_sem_meta_curto', 0.3, 'static', False, 400),
        ('flaml_sem_meta_medio', 1.0, 'static', False, 500),
        ('flaml_ensemble_medio', 1.0, 'data', True, 600),
    ]


def run_all_automl_experiments(force_rerun=False, max_workers=3):
    """
    Executa e faz cache de todos os modos do FLAML nos 56 datasets.
    """
    base_dir = Path(__file__).resolve().parent.parent
    proc_dir = base_dir / 'data' / 'processed'
    features_file = proc_dir / 'image_features_dataset.csv'

    df_all = pd.read_csv(features_file)
    dataset_names = sorted(df_all['dataset_name'].unique().tolist())
    dataset_slices = {ds: df_all[df_all['dataset_name'] == ds].copy() for ds in dataset_names}

    all_dfs = []
    for mode_name, tb_fold, sp_mode, ens, seed in get_automl_modes():
        out_path = proc_dir / f'p4_automl_{mode_name}.csv'
        if out_path.exists() and not force_rerun:
            print(f"[Cache] Carregando {mode_name} de {out_path.name}", flush=True)
            df_mode = pd.read_csv(out_path)
            all_dfs.append(df_mode)
            continue

        print(
            f"\n[FLAML] Executando modo='{mode_name}' | budget_ds={tb_fold*5:.1f}s "
            f"| starting_points={sp_mode} | ensemble={ens}",
            flush=True,
        )
        t0 = time.perf_counter()
        tasks = [
            (ds, dataset_slices[ds], mode_name, tb_fold, sp_mode, ens, seed)
            for ds in dataset_names
        ]

        # Quando ensemble=True, StackingRegressor usa múltiplos threads internamente, então rodamos 1 dataset por vez
        workers = 1 if ens else max_workers
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                records = list(executor.map(_run_flaml_on_dataset, tasks))
        else:
            records = []
            for idx, task in enumerate(tasks, 1):
                records.append(_run_flaml_on_dataset(task))
                if idx % 10 == 0 or idx == len(tasks):
                    print(f"  [{mode_name}] {idx}/{len(tasks)} datasets concluídos...", flush=True)

        df_mode = pd.DataFrame(records).sort_values('dataset').reset_index(drop=True)
        df_mode.to_csv(out_path, index=False)
        elapsed = time.perf_counter() - t0
        print(
            f"  Concluído '{mode_name}' em {elapsed:.1f}s | "
            f"SRCC médio={df_mode['srcc_oof'].mean():.4f} | "
            f"Perda L média={df_mode['loss_oof'].mean():.4f} | "
            f"Tempo médio/DS={df_mode['wall_time_sec'].mean():.2f}s | "
            f"Trials médios={df_mode['n_trials_total'].mean():.1f}",
            flush=True,
        )
        all_dfs.append(df_mode)

    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined.to_csv(proc_dir / 'p4_automl_all_modes.csv', index=False)
    print(f"\nResultados consolidados do FLAML salvos em: {proc_dir / 'p4_automl_all_modes.csv'}")
    return df_combined


if __name__ == '__main__':
    run_all_automl_experiments(force_rerun=False, max_workers=3)
