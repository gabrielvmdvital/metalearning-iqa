import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configurar diretório de cache do matplotlib dentro do workspace para evitar erros de permissão no sandbox
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['MPLCONFIGDIR'] = os.path.join(base_dir, 'data', 'processed', '.matplotlib')

import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr, pearsonr
import scikit_posthocs as sp
import warnings
warnings.filterwarnings('ignore')

from src.hpo import (
    encode_configurations,
    method_1_default,
    method_2_random,
    method_3_smbo_cold,
    method_4_smbo_warm_feurer,
    method_5_smbo_warm_data_independent
)

def run_all_experiments(n_seeds=30, T=20, n_init=5):
    """
    Executa os experimentos completos de HPO sob o protocolo Leave-One-Dataset-Out (LODO).
    """
    print("=" * 70)
    print("INICIANDO EXPERIMENTOS DE HPO (PRÁTICA 3)")
    print(f"Orçamento T={T}, Pontos Iniciais n_init={n_init}, Sementes={n_seeds}")
    print("=" * 70)
    
    proc_dir = Path(base_dir) / 'data' / 'processed'
    fig_dir = Path(base_dir) / 'notebooks'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Carregar matrizes
    loss_matrix = pd.read_csv(proc_dir / 'loss_matrix_H_K_n.csv', index_col=0)
    df_configs = pd.read_csv(proc_dir / 'configurations_K.csv')
    X_base = pd.read_csv(proc_dir / 'X_matrix_base.csv', index_col=0)
    X_ext = pd.read_csv(proc_dir / 'X_matrix_estendida.csv', index_col=0)
    
    datasets = sorted(list(loss_matrix.columns))
    n_datasets = len(datasets)
    K = len(loss_matrix)
    print(f"Dados carregados: {n_datasets} datasets, {K} configurações.")
    
    # Precomputar vetor Z das configurações
    Z = encode_configurations(df_configs)
    
    # Precomputar L* e L_pior por dataset para normalização
    L_star = loss_matrix.min(axis=0)
    L_pior = loss_matrix.max(axis=0)
    L_gap = L_pior - L_star
    L_gap[L_gap < 1e-6] = 1.0 # Evitar divisão por zero caso haja dataset com perda constante
    
    # Definição dos métodos e variantes
    method_names = [
        'M1: Padrão',
        'M2: Random Search',
        'M3: SMBO Cold-Start',
        'M4: Warm-Start (X_ext, k=1)',
        'M4: Warm-Start (X_ext, k=3)',
        'M4: Warm-Start (X_ext, k=5)',
        'M4: Warm-Start (X_base, k=1)',
        'M4: Warm-Start (X_base, k=3)',
        'M4: Warm-Start (X_base, k=5)',
        'M5: Warm-Start (Rank Médio - Controle)'
    ]
    
    # Estruturas para armazenar as curvas e métricas
    # loss_trajectories[method][dt] = array shape (T,) média sobre sementes
    all_trajectories = {m: np.zeros((n_datasets, T)) for m in method_names}
    all_aucs = {m: np.zeros(n_datasets) for m in method_names}
    all_loss_t5 = {m: np.zeros(n_datasets) for m in method_names}
    all_loss_t12 = {m: np.zeros(n_datasets) for m in method_names}
    all_loss_t20 = {m: np.zeros(n_datasets) for m in method_names}
    
    dist_nn_ext = {}
    dist_nn_base = {}
    
    print("\nExecutando simulações LODO em todos os 56 datasets...")
    t0 = time.time()
    
    for d_idx, dt in enumerate(datasets):
        l_star_d = L_star[dt]
        l_gap_d = L_gap[dt]
        
        # M1: Padrão (determinístico)
        _, m1_best = method_1_default(dt, loss_matrix, T=T)
        m1_norm = (np.array(m1_best) - l_star_d) / l_gap_d
        all_trajectories['M1: Padrão'][d_idx] = m1_norm
        all_aucs['M1: Padrão'][d_idx] = np.mean(m1_norm)
        all_loss_t5['M1: Padrão'][d_idx] = m1_norm[n_init - 1]
        all_loss_t12['M1: Padrão'][d_idx] = m1_norm[11]
        all_loss_t20['M1: Padrão'][d_idx] = m1_norm[T - 1]
        
        # M2: Random Search (repetido n_seeds vezes)
        m2_seed_curves = []
        for s in range(n_seeds):
            rng = np.random.RandomState(1000 + s)
            _, m2_best = method_2_random(dt, loss_matrix, T=T, rng=rng)
            m2_seed_curves.append((np.array(m2_best) - l_star_d) / l_gap_d)
        m2_mean_curve = np.mean(m2_seed_curves, axis=0)
        all_trajectories['M2: Random Search'][d_idx] = m2_mean_curve
        all_aucs['M2: Random Search'][d_idx] = np.mean(m2_mean_curve)
        all_loss_t5['M2: Random Search'][d_idx] = m2_mean_curve[n_init - 1]
        all_loss_t12['M2: Random Search'][d_idx] = m2_mean_curve[11]
        all_loss_t20['M2: Random Search'][d_idx] = m2_mean_curve[T - 1]
        
        # M3: SMBO Cold-Start (repetido n_seeds vezes)
        m3_seed_curves = []
        for s in range(n_seeds):
            rng = np.random.RandomState(2000 + s)
            _, m3_best = method_3_smbo_cold(dt, loss_matrix, Z, T=T, n_init=n_init, rng=rng)
            m3_seed_curves.append((np.array(m3_best) - l_star_d) / l_gap_d)
        m3_mean_curve = np.mean(m3_seed_curves, axis=0)
        all_trajectories['M3: SMBO Cold-Start'][d_idx] = m3_mean_curve
        all_aucs['M3: SMBO Cold-Start'][d_idx] = np.mean(m3_mean_curve)
        all_loss_t5['M3: SMBO Cold-Start'][d_idx] = m3_mean_curve[n_init - 1]
        all_loss_t12['M3: SMBO Cold-Start'][d_idx] = m3_mean_curve[11]
        all_loss_t20['M3: SMBO Cold-Start'][d_idx] = m3_mean_curve[T - 1]
        
        # M4: Warm-Start Feurer com variantes (X_ext vs X_base, k in {1, 3, 5})
        for k_val in [1, 3, 5]:
            # X_ext
            m4_ext_curves = []
            cur_dist_ext = 0.0
            for s in range(n_seeds):
                rng = np.random.RandomState(3000 + k_val * 100 + s)
                _, m4_best, cur_dist_ext = method_4_smbo_warm_feurer(
                    dt, loss_matrix, X_ext, Z, k=k_val, T=T, n_init=n_init, rng=rng
                )
                m4_ext_curves.append((np.array(m4_best) - l_star_d) / l_gap_d)
            if k_val == 1:
                dist_nn_ext[dt] = cur_dist_ext
            mname_ext = f'M4: Warm-Start (X_ext, k={k_val})'
            m4_mean_ext = np.mean(m4_ext_curves, axis=0)
            all_trajectories[mname_ext][d_idx] = m4_mean_ext
            all_aucs[mname_ext][d_idx] = np.mean(m4_mean_ext)
            all_loss_t5[mname_ext][d_idx] = m4_mean_ext[n_init - 1]
            all_loss_t12[mname_ext][d_idx] = m4_mean_ext[11]
            all_loss_t20[mname_ext][d_idx] = m4_mean_ext[T - 1]
            
            # X_base
            m4_base_curves = []
            cur_dist_base = 0.0
            for s in range(n_seeds):
                rng = np.random.RandomState(4000 + k_val * 100 + s)
                _, m4_best, cur_dist_base = method_4_smbo_warm_feurer(
                    dt, loss_matrix, X_base, Z, k=k_val, T=T, n_init=n_init, rng=rng
                )
                m4_base_curves.append((np.array(m4_best) - l_star_d) / l_gap_d)
            if k_val == 1:
                dist_nn_base[dt] = cur_dist_base
            mname_base = f'M4: Warm-Start (X_base, k={k_val})'
            m4_mean_base = np.mean(m4_base_curves, axis=0)
            all_trajectories[mname_base][d_idx] = m4_mean_base
            all_aucs[mname_base][d_idx] = np.mean(m4_mean_base)
            all_loss_t5[mname_base][d_idx] = m4_mean_base[n_init - 1]
            all_loss_t12[mname_base][d_idx] = m4_mean_base[11]
            all_loss_t20[mname_base][d_idx] = m4_mean_base[T - 1]
            
        # M5: Warm-Start Data-Independent (Rank Médio Controle)
        m5_curves = []
        for s in range(n_seeds):
            rng = np.random.RandomState(5000 + s)
            _, m5_best = method_5_smbo_warm_data_independent(
                dt, loss_matrix, Z, T=T, n_init=n_init, rng=rng
            )
            m5_curves.append((np.array(m5_best) - l_star_d) / l_gap_d)
        m5_mean = np.mean(m5_curves, axis=0)
        mname_m5 = 'M5: Warm-Start (Rank Médio - Controle)'
        all_trajectories[mname_m5][d_idx] = m5_mean
        all_aucs[mname_m5][d_idx] = np.mean(m5_mean)
        all_loss_t5[mname_m5][d_idx] = m5_mean[n_init - 1]
        all_loss_t12[mname_m5][d_idx] = m5_mean[11]
        all_loss_t20[mname_m5][d_idx] = m5_mean[T - 1]
        
        if (d_idx + 1) % 10 == 0 or (d_idx + 1) == n_datasets:
            print(f"Processados {d_idx + 1}/{n_datasets} datasets...", flush=True)
            
    t1 = time.time()
    print(f"\nSimulações concluídas com sucesso em {t1 - t0:.2f} segundos!", flush=True)
    
    # Resumo Geral e Tabelas
    summary_rows = []
    for m in method_names:
        summary_rows.append({
            'Método': m,
            'AUC (Média)': np.mean(all_aucs[m]),
            'AUC (Desv)': np.std(all_aucs[m]),
            'L_norm(5)': np.mean(all_loss_t5[m]),
            'L_norm(12)': np.mean(all_loss_t12[m]),
            'L_norm(20)': np.mean(all_loss_t20[m])
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(proc_dir / 'hpo_experiment_summary.csv', index=False)
    print("\n--- RESUMO GERAL DAS MÉTRICAS (MÉDIA SOBRE 56 DATASETS) ---")
    print(df_summary.to_string(index=False))
    
    # Salva os resultados por dataset
    per_ds_dict = {'dataset': datasets, 'dist_NN_ext': [dist_nn_ext[d] for d in datasets]}
    for m in method_names:
        per_ds_dict[f'AUC_{m}'] = all_aucs[m]
        per_ds_dict[f'L5_{m}'] = all_loss_t5[m]
        per_ds_dict[f'L20_{m}'] = all_loss_t20[m]
    df_per_ds = pd.DataFrame(per_ds_dict)
    df_per_ds.to_csv(proc_dir / 'hpo_per_dataset_results.csv', index=False)
    
    # Salva as curvas de perda médias ao longo de t
    df_curves = pd.DataFrame({'t': np.arange(1, T + 1)})
    for m in method_names:
        df_curves[m] = np.mean(all_trajectories[m], axis=0)
    df_curves.to_csv(proc_dir / 'hpo_loss_curves_all_methods.csv', index=False)
    
    # Curvas de Perda, Friedman e CD Diagrams
    print("\n--- FIGURA 1: CURVAS DE PERDA NORMALIZADA ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    t_steps = np.arange(1, T + 1)
    
    plot_methods = [
        ('M1: Padrão', '#7f7f7f', ':', 2.0),
        ('M2: Random Search', '#e377c2', '--', 2.0),
        ('M3: SMBO Cold-Start', '#d62728', '-.', 2.5),
        ('M4: Warm-Start (X_ext, k=1)', '#1f77b4', '-', 3.0),
        ('M4: Warm-Start (X_base, k=1)', '#2ca02c', '-', 2.2),
        ('M5: Warm-Start (Rank Médio - Controle)', '#ff7f0e', '--', 2.5)
    ]
    
    for m, col, ls, lw in plot_methods:
        mean_curve = np.mean(all_trajectories[m], axis=0)
        sem_curve = np.std(all_trajectories[m], axis=0) / np.sqrt(n_datasets)
        ax.plot(t_steps, mean_curve, label=m, color=col, linestyle=ls, linewidth=lw)
        ax.fill_between(t_steps, mean_curve - sem_curve, mean_curve + sem_curve, color=col, alpha=0.12)
        
    ax.axvline(x=n_init, color='black', linestyle=':', alpha=0.7, label=f'Fim da Inicialização (t={n_init})')
    ax.set_xlabel('Avaliações de Hiperparâmetros (t)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Perda Normalizada Média $L_{norm}(t)$', fontsize=13, fontweight='bold')
    ax.set_title(r'Convergência da Otimização de Hiperparâmetros (SVR)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(1, T)
    ax.set_ylim(-0.02, 0.70)
    ax.set_xticks(range(1, T + 1))
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right', fontsize=11)
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_1_curvas_perda.png')
    plt.close()
    print(f"Salvo em: {fig_dir / 'fig_1_curvas_perda.png'}")
    
    # Testes Estatísticos de Friedman e CD Diagram
    print("\n--- ANÁLISE ESTATÍSTICA: FRIEDMAN E CD DIAGRAMS ---")
    core_methods = [
        'M1: Padrão',
        'M2: Random Search',
        'M3: SMBO Cold-Start',
        'M4: Warm-Start (X_ext, k=1)',
        'M4: Warm-Start (X_base, k=1)',
        'M5: Warm-Start (Rank Médio - Controle)'
    ]
    short_names = {
        'M1: Padrão': 'M1: Padrão',
        'M2: Random Search': 'M2: Random',
        'M3: SMBO Cold-Start': 'M3: SMBO Cold',
        'M4: Warm-Start (X_ext, k=1)': 'M4: Warm (X_ext)',
        'M4: Warm-Start (X_base, k=1)': 'M4: Warm (X_base)',
        'M5: Warm-Start (Rank Médio - Controle)': 'M5: Rank Médio'
    }
    
    # Friedman nos 3 momentos: t=5, t=12, t=20 e AUC
    eval_moments = [
        ('t=5 (Inicialização)', all_loss_t5),
        ('t=12 (Intermediário)', all_loss_t12),
        ('t=20 (Final)', all_loss_t20),
        ('AUC Global', all_aucs)
    ]
    
    friedman_results = []
    for m_label, m_data in eval_moments:
        arrays = [m_data[m] for m in core_methods]
        stat, pval = friedmanchisquare(*arrays)
        friedman_results.append({
            'Momento': m_label,
            'Friedman Stat': stat,
            'p-value': pval,
            'Significativo (p < 0.05)': pval < 0.05
        })
    df_friedman = pd.DataFrame(friedman_results)
    df_friedman.to_csv(proc_dir / 'hpo_friedman_tests.csv', index=False)
    print("Testes de Friedman:")
    print(df_friedman.to_string(index=False))
    
    # Diagramas de Diferença Crítica nos 3 momentos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    for idx, (m_label, m_data) in enumerate([('t = 5', all_loss_t5), ('t = 12', all_loss_t12), ('t = 20', all_loss_t20)]):
        df_ranks = pd.DataFrame({short_names[m]: m_data[m] for m in core_methods})
        avg_ranks = df_ranks.rank(axis=1, ascending=True).mean(axis=0)
        p_values = sp.posthoc_nemenyi_friedman(df_ranks.values)
        p_values.columns = avg_ranks.index
        p_values.index = avg_ranks.index
        
        plt.sca(axes[idx])
        sp.critical_difference_diagram(avg_ranks, p_values, ax=axes[idx])
        axes[idx].set_title(f"Diagrama CD em {m_label}", fontsize=13, fontweight='bold')
        
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_2_cd_diagrams.png')
    plt.close()
    print(f"Salvo em: {fig_dir / 'fig_2_cd_diagrams.png'}")
    
    # ANÁLISE 2: Papel das Meta-características
    print("\n--- FIGURA 3: O PAPEL DAS META-CARACTERÍSTICAS ---")
    # Comparar k in {1, 3, 5} para X_ext vs X_base vs M5 (Controle)
    k_vals = [1, 3, 5]
    auc_ext = [np.mean(all_aucs[f'M4: Warm-Start (X_ext, k={k})']) for k in k_vals]
    auc_base = [np.mean(all_aucs[f'M4: Warm-Start (X_base, k={k})']) for k in k_vals]
    auc_m5 = np.mean(all_aucs['M5: Warm-Start (Rank Médio - Controle)'])
    auc_m3 = np.mean(all_aucs['M3: SMBO Cold-Start'])
    
    l5_ext = [np.mean(all_loss_t5[f'M4: Warm-Start (X_ext, k={k})']) for k in k_vals]
    l5_base = [np.mean(all_loss_t5[f'M4: Warm-Start (X_base, k={k})']) for k in k_vals]
    l5_m5 = np.mean(all_loss_t5['M5: Warm-Start (Rank Médio - Controle)'])
    l5_m3 = np.mean(all_loss_t5['M3: SMBO Cold-Start'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Gráfico 1: AUC
    x = np.arange(len(k_vals))
    width = 0.35
    ax1.bar(x - width/2, auc_base, width, label='M4: Meta-features Base (16)', color='#2ca02c', alpha=0.85)
    ax1.bar(x + width/2, auc_ext, width, label='M4: Meta-features Estendidas (24)', color='#1f77b4', alpha=0.85)
    ax1.axhline(y=auc_m5, color='#ff7f0e', linestyle='--', linewidth=2.2, label='M5: Rank Médio (Controle Independente)')
    ax1.axhline(y=auc_m3, color='#d62728', linestyle=':', linewidth=2.2, label='M3: SMBO Cold-Start')
    ax1.set_xlabel('Número de Vizinhos Mais Próximos ($k$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'AUC Média da Perda Normalizada (menor é melhor)', fontsize=12, fontweight='bold')
    ax1.set_title('Efeito de $k$ e Espaço de Meta-características na AUC', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'k = {k}' for k in k_vals])
    max_auc_val = max(auc_m3, auc_m5, max(auc_ext), max(auc_base))
    ax1.set_ylim(0, max_auc_val * 1.30)
    ax1.legend(frameon=True, facecolor='white', loc='upper right', fontsize=9.5)
    
    # Gráfico 2: Perda em t=5
    ax2.bar(x - width/2, l5_base, width, label='M4: Meta-features Base (16)', color='#2ca02c', alpha=0.85)
    ax2.bar(x + width/2, l5_ext, width, label='M4: Meta-features Estendidas (24)', color='#1f77b4', alpha=0.85)
    ax2.axhline(y=l5_m5, color='#ff7f0e', linestyle='--', linewidth=2.2, label='M5: Rank Médio (Controle Independente)')
    ax2.axhline(y=l5_m3, color='#d62728', linestyle=':', linewidth=2.2, label='M3: SMBO Cold-Start')
    ax2.set_xlabel('Número de Vizinhos Mais Próximos ($k$)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'Perda em $t=5$ (Fim da Inicialização)', fontsize=12, fontweight='bold')
    ax2.set_title('Qualidade dos Pontos Iniciais em $t = n_{init} = 5$', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'k = {k}' for k in k_vals])
    max_l5_val = max(l5_m3, l5_m5, max(l5_ext), max(l5_base))
    ax2.set_ylim(0, max_l5_val * 1.25)
    ax2.legend(frameon=True, facecolor='white', loc='upper right', fontsize=9.5)
    
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_3_papel_metacaracteristicas.png')
    plt.close()
    print(f"Salvo em: {fig_dir / 'fig_3_papel_metacaracteristicas.png'}")
    
    # Wilcoxon pareado: M4(X_ext, k=1) vs M5, e X_ext vs X_base
    stat_m4_m5, p_m4_m5 = wilcoxon(all_aucs['M4: Warm-Start (X_ext, k=1)'], all_aucs['M5: Warm-Start (Rank Médio - Controle)'])
    stat_ext_base, p_ext_base = wilcoxon(all_aucs['M4: Warm-Start (X_ext, k=1)'], all_aucs['M4: Warm-Start (X_base, k=1)'])
    print(f"\nTeste Wilcoxon Pareado na AUC:")
    print(f"M4 (X_ext, k=1) vs M5 (Rank Médio): p = {p_m4_m5:.5e} (W={stat_m4_m5})")
    print(f"M4 (X_ext, k=1) vs M4 (X_base, k=1): p = {p_ext_base:.5e} (W={stat_ext_base})")
    
    # Quando o Warm-Start Falha
    print("\n--- FIGURA 4: ANÁLISE DE FALHAS DO WARM-START ---")
    # Ganho do Método 4 (X_ext, k=1) sobre o Método 3 (Cold-Start):
    # Ganho = AUC_M3 - AUC_M4. (Como menor AUC é melhor, Ganho > 0 indica que M4 superou M3).
    gain_auc = all_aucs['M3: SMBO Cold-Start'] - all_aucs['M4: Warm-Start (X_ext, k=1)']
    gain_t5 = all_loss_t5['M3: SMBO Cold-Start'] - all_loss_t5['M4: Warm-Start (X_ext, k=1)']
    dists_nn = np.array([dist_nn_ext[d] for d in datasets])
    
    # Correlação entre Ganho e Distância ao Vizinho Mais Próximo
    r_pearson, p_pearson = pearsonr(dists_nn, gain_auc)
    r_spearman, p_spearman = spearmanr(dists_nn, gain_auc)
    print(f"Correlação Ganho AUC vs dist_NN: Pearson r = {r_pearson:.3f} (p={p_pearson:.4f}), Spearman rho = {r_spearman:.3f} (p={p_spearman:.4f})")
    
    df_failures = pd.DataFrame({
        'dataset': datasets,
        'dist_NN': dists_nn,
        'gain_auc': gain_auc,
        'gain_t5': gain_t5,
        'AUC_M3': all_aucs['M3: SMBO Cold-Start'],
        'AUC_M4': all_aucs['M4: Warm-Start (X_ext, k=1)'],
        'failed': gain_auc < 0
    }).sort_values('gain_auc')
    df_failures.to_csv(proc_dir / 'hpo_failures_analysis.csv', index=False)
    
    failed_datasets = df_failures[df_failures['failed']]
    print(f"\nTotal de datasets onde Warm-Start falhou (Ganho AUC < 0): {len(failed_datasets)} de {n_datasets} ({len(failed_datasets)/n_datasets*100:.1f}%)")
    print(failed_datasets[['dataset', 'dist_NN', 'gain_auc', 'gain_t5', 'AUC_M3', 'AUC_M4']].to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    
    colors = ['#d62728' if f else '#1f77b4' for f in df_failures['failed']]
    scatter = ax.scatter(df_failures['dist_NN'], df_failures['gain_auc'], c=colors, s=70, alpha=0.85, edgecolors='k', linewidth=0.6)
    
    # Linha de tendência com proteção
    mask = np.isfinite(df_failures['dist_NN']) & np.isfinite(df_failures['gain_auc'])
    if np.sum(mask) >= 2:
        m_reg, b_reg = np.polyfit(df_failures.loc[mask, 'dist_NN'], df_failures.loc[mask, 'gain_auc'], 1)
        x_line = np.linspace(df_failures.loc[mask, 'dist_NN'].min(), df_failures.loc[mask, 'dist_NN'].max(), 100)
        ax.plot(x_line, m_reg * x_line + b_reg, color='darkblue', linestyle='--', linewidth=2.0, label=f'Tendência Linear ($r={r_pearson:.2f}, p={p_pearson:.3f}$)')
    
    ax.axhline(0, color='red', linestyle=':', linewidth=1.5, label='Limiar de Empate (Ganho = 0)')
    
    # Anotar os principais casos de falha e de maior sucesso
    for _, row in df_failures.head(4).iterrows(): # Piores falhas
        ax.annotate(row['dataset'], (row['dist_NN'], row['gain_auc']),
                    xytext=(5, -12), textcoords='offset points', fontsize=8.5, fontweight='bold', color='#b2182b')
    for _, row in df_failures.tail(4).iterrows(): # Maiores sucessos
        ax.annotate(row['dataset'], (row['dist_NN'], row['gain_auc']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8.5, fontweight='bold', color='#2166ac')
        
    ax.set_xlabel('Distância Euclidiana ao Vizinho Mais Próximo ($dist_{NN}$)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Ganho do Warm-Start na AUC ($\mathrm{AUC}_{M3} - \mathrm{AUC}_{M4}$)', fontsize=12, fontweight='bold')
    ax.set_title(r'Quando o Warm-Start Falha: Ganho de Desempenho vs. Distância ao Vizinho Mais Próximo', fontsize=13, fontweight='bold', pad=12)
    ax.legend(frameon=True, facecolor='white', loc='upper right', fontsize=10.5)
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_4_analise_falhas_scatter.png')
    plt.close()
    print(f"Salvo em: {fig_dir / 'fig_4_analise_falhas_scatter.png'}")
    
    # FIGURA 5: Distribuição de AUC (Boxplots)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    data_to_plot = [all_aucs[m] for m in core_methods]
    labels_to_plot = [short_names[m] for m in core_methods]
    
    sns.boxplot(data=data_to_plot, ax=ax, palette=['#bfbfbf', '#f4cae4', '#fb9a99', '#a6cee3', '#b2df8a', '#fdbf6f'], width=0.55)
    sns.stripplot(data=data_to_plot, ax=ax, color='black', alpha=0.3, jitter=0.2, size=4)
    ax.set_xticklabels(labels_to_plot, rotation=15, ha='right', fontsize=11, fontweight='bold')
    ax.set_ylabel('Área Sob a Curva Normalizada (AUC)', fontsize=12, fontweight='bold')
    ax.set_title('Distribuição da AUC da Perda Normalizada entre os 56 Datasets de Teste', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_5_boxplots_auc.png')
    plt.close()
    print(f"Salvo em: {fig_dir / 'fig_5_boxplots_auc.png'}")
    
    return df_summary

if __name__ == '__main__':
    run_all_experiments(n_seeds=15, T=20, n_init=5)
