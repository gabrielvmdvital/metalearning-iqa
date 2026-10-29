import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def encode_configurations(df_configs):
    """
    Codifica o espaço de configurações Theta (K x 4) em um espaço contínuo normalizado [0, 1]^4
    para ser utilizado como espaço de entrada pelo modelo substituto GP.
    Hiperparâmetros:
    1. log10(C): continuous [-2, 3]
    2. log10(epsilon): continuous [-3, log10(0.5)]
    3. log10(gamma_eff): continuous [-4, 1] (scale/auto mapeado para ~ 1/9 = 0.1111)
    4. is_linear: binary {0, 1}
    """
    Z = []
    for _, row in df_configs.iterrows():
        c_val = np.log10(float(row['C']))
        eps_val = np.log10(float(row['epsilon']))
        
        # Gamma
        g_raw = row['gamma']
        if str(g_raw).lower() in ['scale', 'auto']:
            # Com 9 features padronizadas (var=1), gamma='scale' = 1 / (9 * 1) = 0.1111
            g_eff = 1.0 / 9.0
        else:
            try:
                g_eff = float(g_raw)
            except Exception:
                g_eff = 1.0 / 9.0
        gamma_val = np.log10(g_eff)
        
        is_linear = 1.0 if row['kernel'] == 'linear' else 0.0
        Z.append([c_val, eps_val, gamma_val, is_linear])
        
    Z = np.array(Z)
    scaler = MinMaxScaler()
    Z_norm = scaler.fit_transform(Z)
    return Z_norm

def compute_expected_improvement(mu, sigma, f_best, xi=0.01):
    """
    Expected Improvement (EI) para minimização da função objetivo (perda L).
    EI(x) = (f_best - mu(x) - xi) * Phi(z) + sigma(x) * phi(z)
    onde z = (f_best - mu(x) - xi) / sigma(x)
    """
    ei = np.zeros_like(mu)
    valid = sigma > 1e-8
    
    if np.any(valid):
        diff = f_best - mu[valid] - xi
        z = diff / sigma[valid]
        ei[valid] = diff * norm.cdf(z) + sigma[valid] * norm.pdf(z)
        
    ei[ei < 0] = 0.0
    return ei

class SMBOSimulator:
    """
    Simulador de Sequential Model-Based Optimization (SMBO / Otimização Bayesiana)
    para o espaço discreto K={0, ..., K-1} pré-computado.
    """
    def __init__(self, Z, rng=None):
        self.Z = Z # (K, 4) representações normalizadas
        self.K = len(Z)
        self.rng = rng if rng is not None else np.random.RandomState(42)
        
    def _fit_and_predict(self, evaluated_indices, observed_losses, candidate_indices):
        """
        Ajusta o Gaussian Process nos pontos avaliados e prediz média e desvio padrão
        nos pontos candidatos ainda não avaliados.
        """
        X_train = self.Z[evaluated_indices]
        y_train = np.array(observed_losses)
        X_cand = self.Z[candidate_indices]
        
        # Kernel Matern 5/2 analítico padronizado para espaço Z normalizado em [0, 1]^4
        kernel = Matern(length_scale=0.5, nu=2.5)
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-3,
            optimizer=None,
            normalize_y=True,
            random_state=self.rng.randint(0, 100000)
        )
        
        try:
            gp.fit(X_train, y_train)
            mu, sigma = gp.predict(X_cand, return_std=True)
        except Exception:
            # Fallback se otimizador Cholesky falhar: predição baseada em distância euclidiana simples
            mu = np.mean(y_train) * np.ones(len(candidate_indices))
            sigma = np.std(y_train) * np.ones(len(candidate_indices))
            if np.std(y_train) < 1e-6:
                sigma = 0.1 * np.ones(len(candidate_indices))
                
        return mu, sigma

    def run_smbo_loop(self, true_loss_vector, initial_configs, T=20):
        """
        Executa a busca SMBO partindo dos initial_configs até atingir o orçamento T.
        Retorna:
        - eval_order: lista de configs avaliadas
        - loss_history: perda de cada config avaliada
        - best_history: melhor perda encontrada até cada passo t in [1, T]
        """
        eval_order = list(initial_configs)
        loss_history = [true_loss_vector[c] for c in eval_order]
        
        best_history = []
        cur_best = float('inf')
        for loss in loss_history:
            cur_best = min(cur_best, loss)
            best_history.append(cur_best)
            
        remaining_candidates = set(range(self.K)) - set(eval_order)
        
        while len(eval_order) < T and len(remaining_candidates) > 0:
            candidates_list = sorted(list(remaining_candidates))
            
            # Ajustar GP e calcular Expected Improvement
            mu, sigma = self._fit_and_predict(eval_order, loss_history, candidates_list)
            ei = compute_expected_improvement(mu, sigma, f_best=cur_best, xi=0.01)
            
            # Se todos os EI forem zero ou iguais, escolhe aleatoriamente com desempate
            if np.max(ei) <= 1e-12:
                next_idx = self.rng.choice(candidates_list)
            else:
                best_cand_pos = np.argmax(ei)
                next_idx = candidates_list[best_cand_pos]
                
            loss = true_loss_vector[next_idx]
            eval_order.append(next_idx)
            loss_history.append(loss)
            cur_best = min(cur_best, loss)
            best_history.append(cur_best)
            remaining_candidates.remove(next_idx)
            
        # Se T for maior que as avaliações possíveis, repete o melhor
        while len(best_history) < T:
            best_history.append(cur_best)
            
        return eval_order, loss_history, best_history[:T]

# Implementação dos Métodos de Busca

def method_1_default(dt, loss_matrix, T=20):
    """
    Método 1: Configuração padrão da biblioteca (index 0).
    Referência fixa sem busca.
    """
    l_default = loss_matrix.loc[0, dt]
    best_history = [l_default] * T
    eval_order = [0] * T
    return eval_order, best_history

def method_2_random(dt, loss_matrix, T=20, rng=None):
    """
    Método 2: Random search sobre as K configurações, sem reposição.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    K = len(loss_matrix)
    perm = rng.permutation(K)[:T]
    losses = [loss_matrix.loc[c, dt] for c in perm]
    
    best_history = []
    cur_best = float('inf')
    for l in losses:
        cur_best = min(cur_best, l)
        best_history.append(cur_best)
        
    return list(perm), best_history

def method_3_smbo_cold(dt, loss_matrix, Z, T=20, n_init=5, rng=None):
    """
    Método 3: SMBO com partida fria (cold-start).
    n_init pontos aleatórios sem reposição seguidos de SMBO com GP + EI.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    K = len(loss_matrix)
    initial_configs = rng.choice(K, size=n_init, replace=False).tolist()
    
    simulator = SMBOSimulator(Z, rng=rng)
    true_losses = loss_matrix[dt].values
    eval_order, loss_hist, best_hist = simulator.run_smbo_loop(true_losses, initial_configs, T=T)
    return eval_order, best_hist

def select_feurer_initial_configs(loss_matrix, train_datasets, k_nearest, n_init=5):
    """
    Seleciona n_init configurações usando a estratégia de Feurer et al. (2014):
    Seleciona a melhor configuração do vizinho 1, depois do vizinho 2, ... até o vizinho k.
    Se n_init > k, pega a 2a melhor do vizinho 1, etc., sem duplicatas.
    """
    # Encontra a ordem de melhores configurações para cada um dos k vizinhos
    neighbor_rankings = []
    for d in k_nearest:
        sorted_configs = loss_matrix[d].sort_values(ascending=True).index.tolist()
        neighbor_rankings.append(sorted_configs)
        
    initial_configs = []
    rank_level = 0
    while len(initial_configs) < n_init:
        for n_idx in range(len(k_nearest)):
            if len(initial_configs) >= n_init:
                break
            if rank_level < len(neighbor_rankings[n_idx]):
                cfg = neighbor_rankings[n_idx][rank_level]
                if cfg not in initial_configs:
                    initial_configs.append(cfg)
        rank_level += 1
        if rank_level > 75: # Proteção contra loop infinito
            break
            
    return initial_configs

def method_4_smbo_warm_feurer(dt, loss_matrix, X_meta, Z, k=1, T=20, n_init=5, rng=None):
    """
    Método 4: SMBO com warm-start por datasets similares (Feurer et al., 2014).
    Protocolo LODO: dt é excluído da normalização das meta-características.
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    all_datasets = list(loss_matrix.columns)
    train_datasets = [d for d in all_datasets if d != dt]
    
    # Normalização LODO com imputação robusta ajustada estritamente no conjunto de treino
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_meta.loc[train_datasets].values)
    x_target_imp = imputer.transform(X_meta.loc[[dt]].values)
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_imp)
    x_target_norm = scaler.transform(x_target_imp)
    
    # Distâncias euclidianas no espaço de meta-características
    dists = np.linalg.norm(X_train_norm - x_target_norm, axis=1)
    dist_df = pd.DataFrame({'dataset': train_datasets, 'dist': dists}).sort_values('dist')
    
    k_nearest = dist_df['dataset'].head(k).tolist()
    dist_nn = float(dist_df['dist'].iloc[0]) # Distância ao vizinho mais próximo
    
    initial_configs = select_feurer_initial_configs(loss_matrix, train_datasets, k_nearest, n_init=n_init)
    
    simulator = SMBOSimulator(Z, rng=rng)
    true_losses = loss_matrix[dt].values
    eval_order, loss_hist, best_hist = simulator.run_smbo_loop(true_losses, initial_configs, T=T)
    return eval_order, best_hist, dist_nn

def method_5_smbo_warm_data_independent(dt, loss_matrix, Z, T=20, n_init=5, rng=None):
    """
    Método 5: SMBO com warm-start independente dos dados (controle).
    As n_init configurações iniciais são as de melhor rank médio sobre os datasets de treino (LODO).
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    all_datasets = list(loss_matrix.columns)
    train_datasets = [d for d in all_datasets if d != dt]
    
    # Matriz de perdas de treino
    L_train = loss_matrix[train_datasets]
    # Ranks por dataset (1 = menor perda)
    ranks_train = L_train.rank(axis=0, ascending=True)
    # Rank médio das configurações
    mean_ranks = ranks_train.mean(axis=1).sort_values(ascending=True)
    
    initial_configs = mean_ranks.head(n_init).index.tolist()
    
    simulator = SMBOSimulator(Z, rng=rng)
    true_losses = loss_matrix[dt].values
    eval_order, loss_hist, best_hist = simulator.run_smbo_loop(true_losses, initial_configs, T=T)
    return eval_order, best_hist
