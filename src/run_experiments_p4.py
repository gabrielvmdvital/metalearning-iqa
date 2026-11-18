import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['MPLCONFIGDIR'] = os.path.join(base_dir, 'data', 'processed', '.matplotlib')
sys.path.insert(0, base_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy.stats import friedmanchisquare, pearsonr, rankdata, spearmanr, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut

from src.experiments import calculate_loss_curve
from src.hpo import encode_configurations, method_4_smbo_warm_feurer
from src.models import (
    BaselineRanker,
    HarrisForest,
    MetaRegressor,
    SignificantWinsRanker,
)

warnings.filterwarnings('ignore')


def _prepare_meta_matrices(proc_dir, dataset_names):
    """Carrega e alinha X_base, X_estendida, P_tabular, L_tabular, R_tabular e folds."""
    df_p = pd.read_csv(proc_dir / 'P_matrix_tabular.csv', index_col=0).loc[dataset_names]
    df_l = pd.read_csv(proc_dir / 'L_matrix_tabular.csv', index_col=0).loc[dataset_names]
    df_r = pd.read_csv(proc_dir / 'R_matrix_tabular.csv', index_col=0).loc[dataset_names]
    df_t = pd.read_csv(proc_dir / 'P_matrix_tabular_times.csv', index_col=0).loc[dataset_names]
    df_folds = pd.read_csv(proc_dir / 'P_matrix_tabular_folds.csv')

    df_xb = pd.read_csv(proc_dir / 'X_matrix_base.csv').set_index('dataset').loc[dataset_names]
    df_xe = pd.read_csv(proc_dir / 'X_matrix_estendida.csv').set_index('dataset').loc[dataset_names]

    return df_p, df_l, df_r, df_t, df_folds, df_xb, df_xe


def evaluate_p1p2_on_tabular_lodo(df_p, df_l, df_r, df_t, df_folds, df_xb, df_xe):
    """
    Reexecuta todas as abordagens de seleção de algoritmos das Práticas 1 e 2
    sobre P_tabular sob protocolo LODO estrito (com imputação por mediana dentro do LODO).
    Retorna o resumo agregado e as predições dataset a dataset de cada abordagem.
    """
    datasets = list(df_p.index)
    alg_cols = list(df_p.columns)
    n_ds = len(datasets)
    loo = LeaveOneOut()

    specs = [
        ('AR', 'agg_ar', 'X_estendida'),
        ('MR', 'agg_mr', 'X_estendida'),
        ('SignificantWins', 'agg_sw', 'X_estendida'),
        ('Abordagem1 (P)', 'reg_p', 'X_base'),
        ('Abordagem1 (P)', 'reg_p', 'X_estendida'),
        ('Abordagem2 (R)', 'reg_r', 'X_base'),
        ('Abordagem2 (R)', 'reg_r', 'X_estendida'),
        ('HARRIS (L=0.0)', 'harris_0.0', 'X_base'),
        ('HARRIS (L=0.0)', 'harris_0.0', 'X_estendida'),
        ('HARRIS (L=0.25)', 'harris_0.25', 'X_base'),
        ('HARRIS (L=0.25)', 'harris_0.25', 'X_estendida'),
        ('HARRIS (L=0.5)', 'harris_0.5', 'X_base'),
        ('HARRIS (L=0.5)', 'harris_0.5', 'X_estendida'),
        ('HARRIS (L=0.75)', 'harris_0.75', 'X_base'),
        ('HARRIS (L=0.75)', 'harris_0.75', 'X_estendida'),
        ('HARRIS (L=1.0)', 'harris_1.0', 'X_base'),
        ('HARRIS (L=1.0)', 'harris_1.0', 'X_estendida'),
    ]

    P_val = df_p.values
    L_val = df_l.values
    R_val = df_r.values
    T_val = df_t.values

    per_ds_records = []
    summary_records = []

    for label, kind, x_name in specs:
        X_raw = df_xb.values if x_name == 'X_base' else df_xe.values
        full_name = label if kind.startswith('agg_') else f"{label} ({x_name})"

        losses_top1 = []
        times_top1 = []
        srccs = []
        aucs = []
        rec_algs = []
        fit_times = []
        pred_times = []

        for tr_idx, te_idx in loo.split(X_raw):
            ds_test = datasets[te_idx[0]]
            train_ds = [datasets[i] for i in tr_idx]

            imputer = SimpleImputer(strategy='median')
            X_tr = imputer.fit_transform(X_raw[tr_idx])
            X_te = imputer.transform(X_raw[te_idx])
            P_tr, R_tr = P_val[tr_idx], R_val[tr_idx]

            t0_fit = time.perf_counter()
            if kind == 'agg_ar':
                model = BaselineRanker(method='AR').fit(R_tr)
            elif kind == 'agg_mr':
                model = BaselineRanker(method='MR').fit(R_tr)
            elif kind == 'agg_sw':
                df_f_tr = df_folds[df_folds['dataset'].isin(train_ds)]
                model = SignificantWinsRanker(alpha=0.05).fit(df_f_tr, alg_cols)
            elif kind == 'reg_p':
                model = MetaRegressor(target='P', n_estimators=100, random_state=42).fit(X_tr, P_tr)
            elif kind == 'reg_r':
                model = MetaRegressor(target='R', n_estimators=100, random_state=42).fit(X_tr, R_tr)
            else:
                lam = float(kind.split('_')[1])
                np.random.seed(42)
                model = HarrisForest(lambd=lam, n_estimators=10, max_depth=5).fit(X_tr, P_tr, R_tr)
            fit_times.append(time.perf_counter() - t0_fit)

            t0_pred = time.perf_counter()
            pred_rank = model.predict(X_te)[0]
            pred_times.append(time.perf_counter() - t0_pred)

            best_idx = int(np.argmin(pred_rank))
            rec_alg = alg_cols[best_idx]
            loss_val = float(L_val[te_idx[0], best_idx])
            time_val = float(T_val[te_idx[0], best_idx])

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sc, _ = spearmanr(pred_rank, R_val[te_idx[0]])
            sc = 0.0 if np.isnan(sc) else float(sc)

            _, auc = calculate_loss_curve(pred_rank, P_val[te_idx[0]])

            losses_top1.append(loss_val)
            times_top1.append(time_val)
            srccs.append(sc)
            aucs.append(auc)
            rec_algs.append(rec_alg)

            per_ds_records.append({
                'dataset': ds_test,
                'approach': full_name,
                'recommended_alg': rec_alg,
                'top1_loss': loss_val,
                'top1_srcc': 1.0 - loss_val,
                'top1_fit_time_s': time_val,
                'ranking_srcc': sc,
                'loss_curve_auc': auc,
                'meta_fit_time_s': fit_times[-1],
                'meta_pred_time_s': pred_times[-1],
            })

        summary_records.append({
            'approach': full_name,
            'is_meta_model': not kind.startswith('agg_'),
            'mean_top1_loss': float(np.mean(losses_top1)),
            'std_top1_loss': float(np.std(losses_top1)),
            'mean_top1_srcc': float(1.0 - np.mean(losses_top1)),
            'mean_ranking_srcc': float(np.mean(srccs)),
            'mean_loss_curve_auc': float(np.mean(aucs)),
            'mean_alg_fit_time_s': float(np.mean(times_top1)),
            'mean_meta_fit_time_s': float(np.mean(fit_times)),
            'mean_meta_pred_time_s': float(np.mean(pred_times)),
        })

    df_summary = pd.DataFrame(summary_records).sort_values('mean_top1_loss').reset_index(drop=True)
    df_per_ds = pd.DataFrame(per_ds_records)
    return df_summary, df_per_ds


def compute_warm_start_hpo_losses(proc_dir, datasets, n_seeds=15):
    """
    Calcula a perda real L = 1 - SRCC (não normalizada) do Warm-Start HPO (Prática 3,
    M4: X_estendida, k=1) em t=5 e t=20 sobre os 56 datasets em LODO.
    """
    loss_matrix = pd.read_csv(proc_dir / 'loss_matrix_H_K_n.csv', index_col=0)
    df_configs = pd.read_csv(proc_dir / 'configurations_K.csv')
    X_ext = pd.read_csv(proc_dir / 'X_matrix_estendida.csv', index_col=0)
    Z = encode_configurations(df_configs)

    rows = []
    for ds in datasets:
        l_t5_seeds = []
        l_t20_seeds = []
        dist_nn = 0.0
        for s in range(n_seeds):
            rng = np.random.RandomState(3100 + s)
            _, best_hist, dist_nn = method_4_smbo_warm_feurer(
                ds, loss_matrix, X_ext, Z, k=1, T=20, n_init=5, rng=rng
            )
            l_t5_seeds.append(best_hist[4])
            l_t20_seeds.append(best_hist[19])

        l_star_svr = float(loss_matrix[ds].min())
        l_default_svr = float(loss_matrix.loc[0, ds])
        rows.append({
            'dataset': ds,
            'dist_NN_ext': float(dist_nn),
            'loss_hpo_t5': float(np.mean(l_t5_seeds)),
            'loss_hpo_t20': float(np.mean(l_t20_seeds)),
            'loss_svr_star': l_star_svr,
            'loss_svr_default': l_default_svr,
        })

    return pd.DataFrame(rows).set_index('dataset')


def compute_pyiqa_cross_reference(proc_dir, datasets):
    """
    Calcula referência cruzada com as métricas pré-treinadas de IQA das Práticas 1 e 2
    (P_matrix_folds.csv), separando métricas NR (sem referência) e FR (com referência).
    Usa |SRCC| para lidar com métricas onde menor valor = melhor qualidade (ex.: BRISQUE, LPIPS).
    """
    df_p_folds = pd.read_csv(proc_dir / 'P_matrix_folds.csv')
    df_p_mean = df_p_folds.groupby('dataset').mean(numeric_only=True).reindex(datasets)

    # Remover coluna 'fold' se presente
    metric_cols = [c for c in df_p_mean.columns if c not in ('fold',)]
    nr_cols = [c for c in ['brisque', 'niqe', 'msa_iqa'] if c in metric_cols]
    fr_cols = [c for c in ['ssim', 'psnr', 'lpips'] if c in metric_cols]

    df_abs = df_p_mean[metric_cols].abs()
    best_nr_srcc = df_abs[nr_cols].max(axis=1) if nr_cols else pd.Series(0.0, index=datasets)
    best_all_srcc = df_abs.max(axis=1)

    return pd.DataFrame({
        'pyiqa_best_nr_srcc': best_nr_srcc,
        'pyiqa_best_nr_loss': 1.0 - best_nr_srcc,
        'pyiqa_best_all_srcc': best_all_srcc,
        'pyiqa_best_all_loss': 1.0 - best_all_srcc,
    }, index=datasets)


def run_friedman_and_cd_by_budget(df_master, fig_dir, proc_dir):
    """
    Análise 1: Executa o teste de Friedman e pós-teste de Nemenyi separadamente
    para cada orçamento do AutoML (Curto, Intermediário, Longo) e gera os Diagramas CD.
    """
    budgets = [
        ('Curto (1.5s/DS)', 'FLAML Curto (1.5s)', 'Curto: Budget 1.5s/DS'),
        ('Intermediário (5.0s/DS)', 'FLAML Médio (5.0s)', 'Intermediário: Budget 5.0s/DS'),
        ('Longo (15.0s/DS)', 'FLAML Longo (15.0s)', 'Longo: Budget 15.0s/DS'),
    ]

    base_cols = [
        ('M1: Melhor Médio (AR)', 'M1: Melhor Médio'),
        ('M2: Meta-Learning (LODO)', 'M2: Meta-Learning'),
        ('M3: Warm-Start HPO (T=5)', 'M3: Warm HPO (T=5)'),
        ('M3: Warm-Start HPO (T=20)', 'M3: Warm HPO (T=20)'),
    ]

    friedman_rows = []
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.6), dpi=300)
    plt.subplots_adjust(hspace=0.55)

    for idx, (b_label, flaml_col, plot_title) in enumerate(budgets):
        cols = [c[0] for c in base_cols] + [flaml_col]
        short_map = {
            'M1: Melhor Médio (AR)': 'M1: Melhor Médio',
            'M2: Meta-Learning (LODO)': 'M2: Meta-Learning',
            'M3: Warm-Start HPO (T=5)': 'M3: Warm HPO (T=5)',
            'M3: Warm-Start HPO (T=20)': 'M3: Warm HPO (T=20)',
            flaml_col: f"M4: {flaml_col}",
        }

        df_sub = df_master[cols].copy()
        df_sub.columns = [short_map[c] for c in cols]

        stat, pval = friedmanchisquare(*[df_sub[c].values for c in df_sub.columns])
        avg_ranks = df_sub.rank(axis=1, ascending=True).mean(axis=0)
        p_values = sp.posthoc_nemenyi_friedman(df_sub.values)
        p_values.columns = avg_ranks.index
        p_values.index = avg_ranks.index

        # Comparação pareada direta FLAML vs M2 (Meta-Learning) e M3 (T=20)
        w_m2, p_m2 = wilcoxon(df_master[flaml_col], df_master['M2: Meta-Learning (LODO)'])
        w_m3, p_m3 = wilcoxon(df_master[flaml_col], df_master['M3: Warm-Start HPO (T=20)'])

        friedman_rows.append({
            'Orcamento': b_label,
            'Friedman_Chi2': float(stat),
            'Friedman_pvalue': float(pval),
            'Significativo_0.05': bool(pval < 0.05),
            'Rank_M1_MelhorMedio': float(avg_ranks['M1: Melhor Médio']),
            'Rank_M2_MetaLearning': float(avg_ranks['M2: Meta-Learning']),
            'Rank_M3_WarmHPO_T5': float(avg_ranks['M3: Warm HPO (T=5)']),
            'Rank_M3_WarmHPO_T20': float(avg_ranks['M3: Warm HPO (T=20)']),
            'Rank_M4_FLAML': float(avg_ranks[short_map[flaml_col]]),
            'Wilcoxon_p_FLAML_vs_M2': float(p_m2),
            'Wilcoxon_p_FLAML_vs_M3_T20': float(p_m3),
        })

        plt.sca(axes[idx])
        sp.critical_difference_diagram(avg_ranks, p_values, ax=axes[idx])
        axes[idx].set_title(f"{plot_title}  (Friedman $\\chi^2={stat:.2f}, p={pval:.2e}$)", fontsize=11.5, fontweight='bold', pad=8)

    fig.savefig(fig_dir / 'fig_p4_2_cd_diagrams_orcamentos.png', bbox_inches='tight')
    plt.close()

    df_fried = pd.DataFrame(friedman_rows)
    df_fried.to_csv(proc_dir / 'p4_friedman_tests.csv', index=False)
    return df_fried


def plot_performance_boxplots(df_master, fig_dir):
    """Gera boxplot da perda absoluta L = 1 - SRCC e do Regret relativo ao melhor observado."""
    plot_cols = [
        'M1: Melhor Médio (AR)',
        'M2: Meta-Learning (LODO)',
        'M3: Warm-Start HPO (T=5)',
        'M3: Warm-Start HPO (T=20)',
        'FLAML Curto (1.5s)',
        'FLAML Médio (5.0s)',
        'FLAML Longo (15.0s)',
        'FLAML Sem Meta (5.0s)',
        'FLAML Ensemble (5.0s)',
        'M5: Oráculo (P_tabular)',
    ]
    short_labels = [
        'M1: Melhor\nMédio (AR)',
        'M2: Meta-L.\n(LODO)',
        'M3: Warm\nHPO (T=5)',
        'M3: Warm\nHPO (T=20)',
        'M4: FLAML\nCurto (1.5s)',
        'M4: FLAML\nMédio (5.0s)',
        'M4: FLAML\nLongo (15s)',
        'M4: FLAML\nSem Meta',
        'M4: FLAML\nEnsemble',
        'M5: Oráculo\n(P_tab)',
    ]
    palette = [
        '#9e9e9e', '#4caf50', '#64b5f6', '#1565c0',
        '#ffcc80', '#fb8c00', '#e65100', '#bcaaa4', '#ab47bc', '#263238'
    ]

    best_obs = df_master[plot_cols].min(axis=1)
    df_regret = df_master[plot_cols].sub(best_obs, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.8), dpi=300)

    data_loss = [df_master[c].values for c in plot_cols]
    sns.boxplot(data=data_loss, ax=ax1, palette=palette, width=0.58, showfliers=False)
    sns.stripplot(data=data_loss, ax=ax1, color='black', alpha=0.25, jitter=0.18, size=3.2)
    ax1.axhline(df_master['M5: Oráculo Global'].mean(), color='#d32f2f', linestyle='--', linewidth=1.6, label='Média Oráculo Global ($P_{tab} \\cup K_{75}$)')
    ax1.set_xticklabels(short_labels, rotation=28, ha='right', fontsize=8.8, fontweight='bold')
    ax1.set_ylabel('Perda $L = 1 - \\mathrm{SRCC}$ (menor é melhor)', fontsize=11, fontweight='bold')
    ax1.set_title('Desempenho Absoluto nos 56 Datasets de IQA', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8.5, frameon=True)

    data_reg = [df_regret[c].values for c in plot_cols]
    sns.boxplot(data=data_reg, ax=ax2, palette=palette, width=0.58, showfliers=False)
    sns.stripplot(data=data_reg, ax=ax2, color='black', alpha=0.25, jitter=0.18, size=3.2)
    ax2.set_xticklabels(short_labels, rotation=28, ha='right', fontsize=8.8, fontweight='bold')
    ax2.set_ylabel('Perda Relativa / Regret ($L - L_{\\text{melhor\\_obs}}$)', fontsize=11, fontweight='bold')
    ax2.set_title('Perda Relativa ao Melhor Observado por Dataset', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_p4_1_desempenho_boxplots.png', bbox_inches='tight')
    plt.close()


def plot_cost_vs_performance(df_cost_summary, offline_costs, fig_dir):
    """
    Análise 2: Gera gráfico dual (Desempenho vs. Custo Online em escala log;
    e Custo Acumulado Online+Offline vs. Número de Novos Datasets N).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300)

    # Painel Esquerdo: Desempenho (Perda Média L) vs Custo Online Total (s)
    markers = {
        'M1: Melhor Médio (AR)': ('s', '#757575', 90),
        'M2: Meta-Learning (LODO)': ('^', '#2e7d32', 110),
        'M3: Warm-Start HPO (T=5)': ('D', '#42a5f5', 95),
        'M3: Warm-Start HPO (T=20)': ('D', '#0d47a1', 115),
        'FLAML Curto (1.5s)': ('o', '#ffb74d', 100),
        'FLAML Médio (5.0s)': ('o', '#f57c00', 110),
        'FLAML Longo (15.0s)': ('o', '#e65100', 120),
        'FLAML Sem Meta (5.0s)': ('X', '#8d6e63', 105),
        'FLAML Ensemble (5.0s)': ('P', '#8e24aa', 120),
    }

    for _, row in df_cost_summary.iterrows():
        m = row['Método']
        if m not in markers:
            continue
        mk, col, sz = markers[m]
        x = row['Custo_Online_Total_s']
        y = row['Perda_L_Media']
        ax1.scatter(x, y, marker=mk, color=col, s=sz, edgecolors='black', linewidth=0.8, zorder=4)
        offset_map = {
            'M1: Melhor Médio (AR)': (8, -9),
            'M2: Meta-Learning (LODO)': (8, 5),
            'M3: Warm-Start HPO (T=5)': (8, -3),
            'M3: Warm-Start HPO (T=20)': (8, -3),
            'FLAML Curto (1.5s)': (-25, 9),
            'FLAML Médio (5.0s)': (-15, -16),
            'FLAML Longo (15.0s)': (-82, -14),
            'FLAML Sem Meta (5.0s)': (-45, 9),
            'FLAML Ensemble (5.0s)': (8, -2),
        }
        offset = offset_map.get(m, (6, 5))
        lbl_txt = m.replace(' (LODO)', '').replace(' (AR)', '')
        ax1.annotate(lbl_txt, (x, y), xytext=offset, textcoords='offset points', fontsize=8.0, fontweight='bold', color=col)

    # Conectar trajetória do FLAML (Curto -> Médio -> Longo)
    flaml_pts = df_cost_summary[df_cost_summary['Método'].isin(['FLAML Curto (1.5s)', 'FLAML Médio (5.0s)', 'FLAML Longo (15.0s)'])].sort_values('Custo_Online_Total_s')
    ax1.plot(flaml_pts['Custo_Online_Total_s'], flaml_pts['Perda_L_Media'], linestyle='--', color='#f57c00', alpha=0.65, linewidth=1.8, label='Trajetória FLAML (1.5s → 5s → 15s)')

    ax1.set_xscale('log')
    ax1.set_xlim(3.8, 25.0)
    ax1.set_ylim(0.24, 0.67)
    ax1.set_xlabel('Custo Online Médio por Dataset (s) — Escala Log', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Perda Média $L = 1 - \\mathrm{SRCC}$ (menor é melhor)', fontsize=11, fontweight='bold')
    ax1.set_title('Fronteira Desempenho vs. Custo Online por Dataset', fontsize=12, fontweight='bold')
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)
    ax1.legend(loc='center right', fontsize=8.5, frameon=True)

    # Painel Direito: Custo Acumulado (Offline + N * Online) vs N novos datasets
    N_vals = np.linspace(1, 200, 200)
    c_on_m2 = float(df_cost_summary.loc[df_cost_summary['Método'] == 'M2: Meta-Learning (LODO)', 'Custo_Online_Total_s'].iloc[0])
    c_on_m3_t20 = float(df_cost_summary.loc[df_cost_summary['Método'] == 'M3: Warm-Start HPO (T=20)', 'Custo_Online_Total_s'].iloc[0])
    c_on_flaml_m = float(df_cost_summary.loc[df_cost_summary['Método'] == 'FLAML Médio (5.0s)', 'Custo_Online_Total_s'].iloc[0])
    c_on_flaml_l = float(df_cost_summary.loc[df_cost_summary['Método'] == 'FLAML Longo (15.0s)', 'Custo_Online_Total_s'].iloc[0])

    c_off_m2 = offline_costs['offline_m2_total_s']
    c_off_m3 = offline_costs['offline_m3_total_s']

    ax2.plot(N_vals, c_off_m2 + N_vals * c_on_m2, color='#2e7d32', linewidth=2.2, label=f'M2 Meta-Learning (Off={c_off_m2:.0f}s + {c_on_m2:.2f}s/DS)')
    ax2.plot(N_vals, c_off_m3 + N_vals * c_on_m3_t20, color='#0d47a1', linewidth=2.2, label=f'M3 Warm-Start T=20 (Off={c_off_m3:.0f}s + {c_on_m3_t20:.2f}s/DS)')
    ax2.plot(N_vals, N_vals * c_on_flaml_m, color='#f57c00', linestyle='--', linewidth=2.0, label=f'FLAML Médio (Off=0s + {c_on_flaml_m:.2f}s/DS)')
    ax2.plot(N_vals, N_vals * c_on_flaml_l, color='#e65100', linestyle='-.', linewidth=2.0, label=f'FLAML Longo (Off=0s + {c_on_flaml_l:.2f}s/DS)')

    # Marcar break-even M3 vs FLAML Longo e M2 vs FLAML Médio
    for c_off, c_meta, c_auto, lbl, col in [
        (c_off_m2, c_on_m2, c_on_flaml_m, 'M2 vs FLAML Médio', '#2e7d32'),
        (c_off_m3, c_on_m3_t20, c_on_flaml_l, 'M3 vs FLAML Longo', '#0d47a1'),
    ]:
        if c_auto > c_meta:
            n_be = c_off / (c_auto - c_meta)
            if n_be <= 200:
                y_be = c_off + n_be * c_meta
                ax2.scatter([n_be], [y_be], color=col, s=70, zorder=5, edgecolors='black')
                ax2.annotate(f'N*={int(np.ceil(n_be))}', (n_be, y_be), xytext=(5, 7), textcoords='offset points', fontsize=8.5, fontweight='bold', color=col)

    ax2.set_xlabel('Número de Novos Datasets ($N_{\\text{novos}}$)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Custo Computacional Acumulado (s)', fontsize=11, fontweight='bold')
    ax2.set_title('Amortização do Custo Offline vs. AutoML ($N_{\\text{novos}}$)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=8.2)

    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_p4_3_desempenho_vs_custo.png', bbox_inches='tight')
    plt.close()


def plot_automl_choices_and_meta(df_automl_all, df_choices, df_p_tab, df_xe, fig_dir):
    """
    Análise 3: Gera fig_p4_4 (escolhas de algoritmos e ablação meta/ensemble) e
    fig_p4_5 (ganho do AutoML vs dist_NN, meta-características e perda do meta-modelo).
    """
    # Figura 4: Frequência de algoritmos escolhidos por orçamento + comparação de ablação
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6), dpi=300)

    modes_order = ['flaml_curto', 'flaml_medio', 'flaml_longo']
    mode_labels = {'flaml_curto': 'FLAML Curto (1.5s)', 'flaml_medio': 'FLAML Médio (5.0s)', 'flaml_longo': 'FLAML Longo (15.0s)'}
    all_learners = ['extra_tree', 'lgbm', 'rf', 'xgboost', 'xgb_limitdepth', 'sgd']

    freq_data = []
    for m in modes_order:
        sub = df_automl_all[df_automl_all['mode'] == m]
        # Contagem por fold (56 * 5 = 280 folds por orçamento)
        fold_ests = []
        for s in sub['best_estimators_folds']:
            fold_ests.extend(json.loads(s))
        cnt = Counter(fold_ests)
        total = max(1, len(fold_ests))
        row = {'Orçamento': mode_labels[m]}
        for lrn in all_learners:
            row[lrn] = 100.0 * cnt.get(lrn, 0) / total
        freq_data.append(row)

    df_freq = pd.DataFrame(freq_data).set_index('Orçamento')
    learner_colors = ['#1565c0', '#2e7d32', '#66bb6a', '#ef6c00', '#ffa726', '#8d6e63']
    df_freq.plot(kind='bar', stacked=True, ax=ax1, color=learner_colors, edgecolor='white', width=0.6)
    ax1.set_ylabel('Percentual de Folds Escolhidos (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Algoritmos Selecionados pelo FLAML (280 folds/orçamento)', fontsize=11.5, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=12, ha='right', fontsize=9.5, fontweight='bold')
    ax1.legend(title='Learner (Azul/Verde=em P_tab | Laranja/Marrom=Fora)', bbox_to_anchor=(0.5, -0.24), loc='upper center', ncol=3, fontsize=8.2)

    # Painel Direito: Decomposição das ablações no orçamento Médio (5.0s)
    ablation_modes = [
        ('flaml_curto', 'FLAML Curto\n(Meta ON, 1.5s)'),
        ('flaml_sem_meta_curto', 'FLAML Curto\n(Meta OFF, 1.5s)'),
        ('flaml_medio', 'FLAML Médio\n(Meta ON, 5.0s)'),
        ('flaml_sem_meta_medio', 'FLAML Médio\n(Meta OFF, 5.0s)'),
        ('flaml_ensemble_medio', 'FLAML Médio\n(Ensemble, 5.0s)'),
        ('flaml_longo', 'FLAML Longo\n(Meta ON, 15.0s)'),
    ]
    ab_means = []
    ab_sems = []
    ab_names = []
    for m_key, m_lbl in ablation_modes:
        vals = df_automl_all[df_automl_all['mode'] == m_key]['loss_oof'].values
        ab_means.append(np.mean(vals))
        ab_sems.append(np.std(vals) / np.sqrt(len(vals)))
        ab_names.append(m_lbl)

    ab_colors = ['#ffb74d', '#d7ccc8', '#fb8c00', '#a1887f', '#8e24aa', '#e65100']
    x_pos = np.arange(len(ab_names))
    bars = ax2.bar(x_pos, ab_means, yerr=ab_sems, capsize=4, color=ab_colors, edgecolor='black', alpha=0.9)
    for bar, val, sem in zip(bars, ab_means, ab_sems):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + sem + 0.012, f'{val:.3f}', ha='center', va='bottom', fontsize=8.8, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ab_names, rotation=20, ha='right', fontsize=8.5, fontweight='bold')
    ax2.set_ylabel('Perda Média $L = 1 - \\mathrm{SRCC}$', fontsize=11, fontweight='bold')
    ax2.set_title('Ablação Interna do FLAML: Meta-Conhecimento e Ensemble', fontsize=11.5, fontweight='bold')
    ax2.set_ylim(0, max(ab_means) * 1.26)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_p4_4_escolhas_automl.png', bbox_inches='tight')
    plt.close()

    # Figura 5: Ganho do FLAML Longo vs M2 (Meta-Learning) e M3 (Warm HPO) vs conhecimento prévio
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=300)

    df_plot = df_choices.copy()
    # Painel (a): Ganho do FLAML Longo sobre Warm-Start HPO (T=20) vs dist_NN (log1p para lidar com ChallengeDB)
    x_dist = np.log1p(df_plot['dist_NN_ext'].values)
    y_gain_hpo = df_plot['gain_flaml_longo_vs_hpo20'].values
    rho1, p1 = spearmanr(df_plot['dist_NN_ext'].values, y_gain_hpo)
    c1 = ['#2e7d32' if g > 0 else '#c62828' for g in y_gain_hpo]
    axes[0].scatter(x_dist, y_gain_hpo, c=c1, s=55, alpha=0.85, edgecolors='black', linewidth=0.5)
    axes[0].axhline(0, color='black', linestyle=':', linewidth=1.2)
    m_fit, b_fit = np.polyfit(x_dist, y_gain_hpo, 1)
    xs = np.linspace(x_dist.min(), x_dist.max(), 100)
    axes[0].plot(xs, m_fit * xs + b_fit, color='#1565c0', linestyle='--', linewidth=1.8, label=f'Spearman $\\rho={rho1:.3f}$ ($p={p1:.3f}$)')
    axes[0].set_xlabel('$\\log(1 + dist_{NN})$ (Atividade 3)', fontsize=10.5, fontweight='bold')
    axes[0].set_ylabel('Ganho FLAML Longo vs. Warm HPO ($L_{M3} - L_{M4}$)', fontsize=10, fontweight='bold')
    axes[0].set_title('(a) Ganho vs. Distância ao Vizinho ($dist_{NN}$)', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # Painel (b): Ganho do FLAML Longo sobre M2 vs NumberOfInstances (tamanho do dataset)
    x_n = np.log10(df_xe.loc[df_plot['dataset'], 'NumberOfInstances'].values)
    y_gain_m2 = df_plot['gain_flaml_longo_vs_m2'].values
    rho2, p2 = spearmanr(x_n, y_gain_m2)
    c2 = ['#2e7d32' if g > 0 else '#c62828' for g in y_gain_m2]
    axes[1].scatter(x_n, y_gain_m2, c=c2, s=55, alpha=0.85, edgecolors='black', linewidth=0.5)
    axes[1].axhline(0, color='black', linestyle=':', linewidth=1.2)
    m_fit2, b_fit2 = np.polyfit(x_n, y_gain_m2, 1)
    xs2 = np.linspace(x_n.min(), x_n.max(), 100)
    axes[1].plot(xs2, m_fit2 * xs2 + b_fit2, color='#1565c0', linestyle='--', linewidth=1.8, label=f'Spearman $\\rho={rho2:.3f}$ ($p={p2:.3f}$)')
    axes[1].set_xlabel('$\\log_{10}(\\text{NumberOfInstances})$ ($X_{est}$)', fontsize=10.5, fontweight='bold')
    axes[1].set_ylabel('Ganho FLAML Longo vs. M2 ($L_{M2} - L_{M4}$)', fontsize=10, fontweight='bold')
    axes[1].set_title('(b) Ganho vs. Tamanho do Dataset ($N$)', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, linestyle=':', alpha=0.5)

    # Painel (c): Ganho do FLAML Longo sobre M2 vs IQA_MeanBlurExtent (meta-feature de domínio IQA)
    x_blur = np.log1p(df_xe.loc[df_plot['dataset'], 'IQA_MeanBlurExtent'].fillna(df_xe['IQA_MeanBlurExtent'].median()).values)
    rho3, p3 = spearmanr(x_blur, y_gain_m2)
    axes[2].scatter(x_blur, y_gain_m2, c=c2, s=55, alpha=0.85, edgecolors='black', linewidth=0.5)
    axes[2].axhline(0, color='black', linestyle=':', linewidth=1.2)
    m_fit3, b_fit3 = np.polyfit(x_blur, y_gain_m2, 1)
    xs3 = np.linspace(x_blur.min(), x_blur.max(), 100)
    axes[2].plot(xs3, m_fit3 * xs3 + b_fit3, color='#1565c0', linestyle='--', linewidth=1.8, label=f'Spearman $\\rho={rho3:.3f}$ ($p={p3:.4f}$)')
    axes[2].set_xlabel('$\\log(1 + \\text{IQA\\_MeanBlurExtent})$ ($X_{est}$)', fontsize=10.5, fontweight='bold')
    axes[2].set_ylabel('Ganho FLAML Longo vs. M2 ($L_{M2} - L_{M4}$)', fontsize=10, fontweight='bold')
    axes[2].set_title('(c) Ganho vs. Nitidez/Laplaciano (Domínio IQA)', fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=8.5)
    axes[2].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_p4_5_ganho_vs_metaconhecimento.png', bbox_inches='tight')
    plt.close()


def run_all_p4_analyses():
    """Orquestra todas as análises da Prática 4 e salva tabelas e figuras."""
    root = Path(base_dir)
    proc_dir = root / 'data' / 'processed'
    fig_dir = root / 'notebooks'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("INICIANDO ANÁLISES DA PRÁTICA 4: AutoML (FLAML) vs. META-APRENDIZAGEM")
    print("=" * 75)

    # 1. Carregar dados base e matrizes P_tabular
    df_p_tab = pd.read_csv(proc_dir / 'P_matrix_tabular.csv', index_col=0)
    datasets = sorted(df_p_tab.index.tolist())
    df_p, df_l, df_r, df_t, df_folds, df_xb, df_xe = _prepare_meta_matrices(proc_dir, datasets)

    # 2. Avaliar abordagens das Práticas 1 e 2 sobre P_tabular em LODO (com cache)
    p1p2_sum_path = proc_dir / 'p4_metalearning_p1p2_tabular.csv'
    p1p2_ds_path = proc_dir / 'p4_metalearning_p1p2_per_ds.csv'
    if p1p2_sum_path.exists() and p1p2_ds_path.exists():
        print("\n[1/5] Carregando abordagens P1/P2 em P_tabular (LODO) do cache...")
        df_p1p2_summary = pd.read_csv(p1p2_sum_path)
        df_p1p2_per_ds = pd.read_csv(p1p2_ds_path)
    else:
        print("\n[1/5] Avaliando abordagens P1/P2 em P_tabular (LODO)...")
        df_p1p2_summary, df_p1p2_per_ds = evaluate_p1p2_on_tabular_lodo(
            df_p, df_l, df_r, df_t, df_folds, df_xb, df_xe
        )
        df_p1p2_summary.to_csv(p1p2_sum_path, index=False)
        df_p1p2_per_ds.to_csv(p1p2_ds_path, index=False)
    print(df_p1p2_summary[['approach', 'mean_top1_loss', 'mean_top1_srcc', 'mean_ranking_srcc', 'mean_loss_curve_auc']].head(8).to_string(index=False))

    # Identificar M1 (AR) e M2 (melhor abordagem treinada com X_base ou X_estendida)
    meta_only = df_p1p2_summary[df_p1p2_summary['is_meta_model']]
    best_m2_name = str(meta_only.iloc[0]['approach'])
    print(f"\nMelhor abordagem P1/P2 selecionada como M2: {best_m2_name}")

    df_m1_ds = df_p1p2_per_ds[df_p1p2_per_ds['approach'] == 'AR'].set_index('dataset').loc[datasets]
    df_m2_ds = df_p1p2_per_ds[df_p1p2_per_ds['approach'] == best_m2_name].set_index('dataset').loc[datasets]

    # 3. Avaliar Warm-Start HPO (Prática 3) em perda real L = 1 - SRCC
    print("\n[2/5] Calculando perdas reais do Warm-Start HPO (Prática 3)...")
    df_hpo = compute_warm_start_hpo_losses(proc_dir, datasets, n_seeds=15)

    # 4. Carregar resultados do FLAML e custos online medidos
    df_automl_all = pd.read_csv(proc_dir / 'p4_automl_all_modes.csv')
    df_costs = pd.read_csv(proc_dir / 'p4_online_costs_measured.csv').set_index('dataset').loc[datasets]
    df_pyiqa = compute_pyiqa_cross_reference(proc_dir, datasets)

    # 5. Consolidar tabela mestre por dataset
    automl_modes_map = {
        'flaml_curto': 'FLAML Curto (1.5s)',
        'flaml_medio': 'FLAML Médio (5.0s)',
        'flaml_longo': 'FLAML Longo (15.0s)',
        'flaml_sem_meta_curto': 'FLAML Sem Meta (1.5s)',
        'flaml_sem_meta_medio': 'FLAML Sem Meta (5.0s)',
        'flaml_ensemble_medio': 'FLAML Ensemble (5.0s)',
    }

    df_master = pd.DataFrame(index=datasets)
    df_master.index.name = 'dataset'
    df_master['M1: Melhor Médio (AR)'] = df_m1_ds['top1_loss']
    df_master['M2: Meta-Learning (LODO)'] = df_m2_ds['top1_loss']
    df_master['M3: Warm-Start HPO (T=5)'] = df_hpo['loss_hpo_t5']
    df_master['M3: Warm-Start HPO (T=20)'] = df_hpo['loss_hpo_t20']

    automl_slices = {}
    for mode_key, col_label in automl_modes_map.items():
        sub = df_automl_all[df_automl_all['mode'] == mode_key].set_index('dataset').loc[datasets]
        automl_slices[mode_key] = sub
        df_master[col_label] = sub['loss_oof']

    df_master['M5: Oráculo (P_tabular)'] = df_l.min(axis=1)
    df_master['M5: Oráculo Global'] = np.minimum(df_l.min(axis=1), df_hpo['loss_svr_star'])
    df_master['PyIQA Best NR (P1/P2)'] = df_pyiqa['pyiqa_best_nr_loss']
    df_master['PyIQA Best All (P1/P2)'] = df_pyiqa['pyiqa_best_all_loss']

    # Tempos online por dataset
    df_master['time_m1_s'] = df_costs['t_img_features_s'] + df_m1_ds['top1_fit_time_s']
    df_master['time_m2_s'] = df_costs['t_img_features_s'] + df_m2_ds['meta_pred_time_s'] + df_m2_ds['top1_fit_time_s']
    df_master['time_m3_t5_s'] = df_costs['t_online_hpo5_total_s']
    df_master['time_m3_t20_s'] = df_costs['t_online_hpo20_total_s']
    for mode_key in automl_modes_map:
        df_master[f'time_{mode_key}_s'] = df_costs['t_img_features_s'] + automl_slices[mode_key]['wall_time_sec']
        df_master[f'wall_{mode_key}_s'] = automl_slices[mode_key]['wall_time_sec']

    df_master.to_csv(proc_dir / 'p4_per_dataset_results.csv')

    # 6. Análise 1: Desempenho + Friedman + Nemenyi CD Diagrams
    print("\n[3/5] Executando Análise 1 (Desempenho, Friedman e Nemenyi CD)...")
    df_fried = run_friedman_and_cd_by_budget(df_master, fig_dir, proc_dir)
    print(df_fried.to_string(index=False))
    plot_performance_boxplots(df_master, fig_dir)

    # 7. Análise 2: Desempenho vs Custo (Online vs Offline + Break-even)
    print("\n[4/5] Executando Análise 2 (Desempenho vs Custo Online/Offline)...")
    loss_cols = [
        'M1: Melhor Médio (AR)',
        'M2: Meta-Learning (LODO)',
        'M3: Warm-Start HPO (T=5)',
        'M3: Warm-Start HPO (T=20)',
        'FLAML Curto (1.5s)',
        'FLAML Médio (5.0s)',
        'FLAML Longo (15.0s)',
        'FLAML Sem Meta (1.5s)',
        'FLAML Sem Meta (5.0s)',
        'FLAML Ensemble (5.0s)',
        'M5: Oráculo (P_tabular)',
        'M5: Oráculo Global',
    ]
    time_col_map = {
        'M1: Melhor Médio (AR)': ('time_m1_s', df_m1_ds['top1_fit_time_s'].values),
        'M2: Meta-Learning (LODO)': ('time_m2_s', (df_m2_ds['meta_pred_time_s'] + df_m2_ds['top1_fit_time_s']).values),
        'M3: Warm-Start HPO (T=5)': ('time_m3_t5_s', (df_costs['t_knn_meta_s'] + df_costs['t_hpo_fit_5_s']).values),
        'M3: Warm-Start HPO (T=20)': ('time_m3_t20_s', (df_costs['t_knn_meta_s'] + df_costs['t_smbo_overhead_20_s'] + df_costs['t_hpo_fit_20_s']).values),
        'FLAML Curto (1.5s)': ('time_flaml_curto_s', df_master['wall_flaml_curto_s'].values),
        'FLAML Médio (5.0s)': ('time_flaml_medio_s', df_master['wall_flaml_medio_s'].values),
        'FLAML Longo (15.0s)': ('time_flaml_longo_s', df_master['wall_flaml_longo_s'].values),
        'FLAML Sem Meta (1.5s)': ('time_flaml_sem_meta_curto_s', df_master['wall_flaml_sem_meta_curto_s'].values),
        'FLAML Sem Meta (5.0s)': ('time_flaml_sem_meta_medio_s', df_master['wall_flaml_sem_meta_medio_s'].values),
        'FLAML Ensemble (5.0s)': ('time_flaml_ensemble_medio_s', df_master['wall_flaml_ensemble_medio_s'].values),
    }

    best_obs_pred = df_master[loss_cols[:-2]].min(axis=1)
    summary_rows = []
    for m_col in loss_cols:
        vals = df_master[m_col].values
        regret_obs = vals - best_obs_pred.values
        regret_ora = vals - df_master['M5: Oráculo Global'].values
        t_tot = float(df_master[time_col_map[m_col][0]].mean()) if m_col in time_col_map else np.nan
        t_mdl = float(np.mean(time_col_map[m_col][1])) if m_col in time_col_map else np.nan
        summary_rows.append({
            'Método': m_col,
            'SRCC_Medio': float(1.0 - np.mean(vals)),
            'Perda_L_Media': float(np.mean(vals)),
            'Perda_L_Desv': float(np.std(vals)),
            'Regret_MelhorObs': float(np.mean(regret_obs)),
            'Regret_OraculoGlobal': float(np.mean(regret_ora)),
            'Custo_Online_Modelo_s': t_mdl,
            'Custo_Online_Total_s': t_tot,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(proc_dir / 'p4_experiment_summary.csv', index=False)
    print(df_summary.round(4).to_string(index=False))

    # Custos offline exatos (55 datasets de treino = 55/56 do total medido)
    c_feat_55 = float(df_costs['t_img_features_s'].sum()) * (55.0 / 56.0)
    c_ptab_55 = float(df_t.sum().sum()) * (55.0 / 56.0)
    c_meta_fit = float(df_m2_ds['meta_fit_time_s'].mean())
    c_hpo_mat_55 = 158.33 * (55.0 / 56.0)  # Medido em build_loss_matrix.py (75 x 56 x 5 SVR fits)

    offline_costs = {
        'offline_feat_55_s': c_feat_55,
        'offline_ptab_55_s': c_ptab_55,
        'offline_meta_fit_s': c_meta_fit,
        'offline_m2_total_s': c_feat_55 + c_ptab_55 + c_meta_fit,
        'offline_hpo_matrix_55_s': c_hpo_mat_55,
        'offline_m3_total_s': c_feat_55 + c_hpo_mat_55,
    }
    with open(proc_dir / 'p4_offline_costs.json', 'w', encoding='utf-8') as f:
        json.dump(offline_costs, f, indent=2)
    print("\nCustos Offline (s):", {k: round(v, 2) for k, v in offline_costs.items()})

    plot_cost_vs_performance(df_summary, offline_costs, fig_dir)

    # 8. Análise 3: O que o AutoML escolheu + Correlações com conhecimento prévio
    print("\n[5/5] Executando Análise 3 (Inspeção Caixa-Branca do AutoML)...")
    ptab_flaml_names = {'extra_tree': 'ExtraTrees', 'rf': 'RandomForest', 'lgbm': 'LightGBM'}
    m2_to_flaml = {'ExtraTrees': 'extra_tree', 'RandomForest': 'rf', 'LightGBM': 'lgbm', 'SVR': 'svr', 'kNN': 'kneighbor', 'MLP': 'mlp'}

    choices_rows = []
    for ds in datasets:
        sub_longo = automl_slices['flaml_longo'].loc[ds]
        sub_medio = automl_slices['flaml_medio'].loc[ds]
        sub_curto = automl_slices['flaml_curto'].loc[ds]
        sub_ens = automl_slices['flaml_ensemble_medio'].loc[ds]

        modal_longo = sub_longo['best_estimator_modal']
        m2_alg = df_m2_ds.loc[ds, 'recommended_alg']
        m2_flaml_equiv = m2_to_flaml.get(m2_alg, m2_alg.lower())
        in_p_tab = modal_longo in ptab_flaml_names
        agrees_m2 = (modal_longo == m2_flaml_equiv)

        # Melhor algoritmo real em P_tabular para esse dataset
        oracle_alg_ptab = df_l.loc[ds].idxmin()

        choices_rows.append({
            'dataset': ds,
            'n_samples': int(sub_longo['n_samples']),
            'oracle_alg_ptab': oracle_alg_ptab,
            'm2_recommended_alg': m2_alg,
            'flaml_curto_modal': sub_curto['best_estimator_modal'],
            'flaml_medio_modal': sub_medio['best_estimator_modal'],
            'flaml_longo_modal': modal_longo,
            'flaml_longo_in_ptab': in_p_tab,
            'flaml_longo_agrees_m2': agrees_m2,
            'flaml_longo_trials': int(sub_longo['n_trials_total']),
            'flaml_ensemble_members_f0': json.loads(sub_ens['ensemble_members_folds'])[0],
            'loss_m2': float(df_master.loc[ds, 'M2: Meta-Learning (LODO)']),
            'loss_m3_t20': float(df_master.loc[ds, 'M3: Warm-Start HPO (T=20)']),
            'loss_flaml_curto': float(df_master.loc[ds, 'FLAML Curto (1.5s)']),
            'loss_flaml_medio': float(df_master.loc[ds, 'FLAML Médio (5.0s)']),
            'loss_flaml_longo': float(df_master.loc[ds, 'FLAML Longo (15.0s)']),
            'loss_flaml_sem_meta_medio': float(df_master.loc[ds, 'FLAML Sem Meta (5.0s)']),
            'loss_flaml_ensemble_medio': float(df_master.loc[ds, 'FLAML Ensemble (5.0s)']),
            'gain_flaml_longo_vs_m2': float(df_master.loc[ds, 'M2: Meta-Learning (LODO)'] - df_master.loc[ds, 'FLAML Longo (15.0s)']),
            'gain_flaml_longo_vs_hpo20': float(df_master.loc[ds, 'M3: Warm-Start HPO (T=20)'] - df_master.loc[ds, 'FLAML Longo (15.0s)']),
            'dist_NN_ext': float(df_hpo.loc[ds, 'dist_NN_ext']),
            'm2_ranking_srcc': float(df_m2_ds.loc[ds, 'ranking_srcc']),
        })

    df_choices = pd.DataFrame(choices_rows)
    df_choices.to_csv(proc_dir / 'p4_automl_choices_analysis.csv', index=False)

    print(f"Taxa de pertencimento a P_tabular (FLAML Longo modal): {df_choices['flaml_longo_in_ptab'].mean()*100:.1f}%")
    print(f"Taxa de concordância FLAML Longo modal == M2: {df_choices['flaml_longo_agrees_m2'].mean()*100:.1f}%")
    print("\nDistribuição modal FLAML Longo:")
    print(df_choices['flaml_longo_modal'].value_counts().to_string())

    # Correlações com meta-conhecimento prévio
    corr_rows = []
    meta_feats_check = [
        'NumberOfInstances', 'IQA_MeanBrightness', 'IQA_MeanContrast',
        'IQA_MeanColorfulness', 'IQA_MeanSpatialFreq', 'IQA_MeanBlurExtent',
        'TargetCoV', 'MeanCorrWithTarget', 'MaxCorrWithTarget', 'LinearRegressionMAE'
    ]
    for target_col in ['gain_flaml_longo_vs_m2', 'gain_flaml_longo_vs_hpo20']:
        rho_d, p_d = spearmanr(df_choices['dist_NN_ext'], df_choices[target_col])
        corr_rows.append({'Alvo': target_col, 'Variavel': 'dist_NN_ext', 'Spearman_rho': rho_d, 'p_value': p_d})
        rho_l, p_l = spearmanr(df_choices['loss_m2'], df_choices[target_col])
        corr_rows.append({'Alvo': target_col, 'Variavel': 'loss_m2 (P1/P2)', 'Spearman_rho': rho_l, 'p_value': p_l})
        for mf in meta_feats_check:
            vals_mf = df_xe.loc[datasets, mf].fillna(df_xe[mf].median()).values
            rho_m, p_m = spearmanr(vals_mf, df_choices[target_col].values)
            corr_rows.append({'Alvo': target_col, 'Variavel': mf, 'Spearman_rho': rho_m, 'p_value': p_m})

    df_corrs = pd.DataFrame(corr_rows)
    df_corrs.to_csv(proc_dir / 'p4_correlations_meta.csv', index=False)

    plot_automl_choices_and_meta(df_automl_all, df_choices, df_p, df_xe, fig_dir)
    print("\nPipeline de análises da Prática 4 concluído com sucesso!")


if __name__ == '__main__':
    run_all_p4_analyses()
