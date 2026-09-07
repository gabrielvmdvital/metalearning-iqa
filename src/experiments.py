# src/experiments.py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneOut
from collections import defaultdict
from tqdm import tqdm

def calculate_loss_curve(predicted_ranks, true_P_row):
    """
    Calcula a Curva de Perda e AUC para uma rodada de teste.
    predicted_ranks: array com o ranking predito (ex: [3, 1, 2] significa que o alg2 é o 1º)
    true_P_row: array com a correlação de Spearman verdadeira dos algoritmos no dataset de teste
    
    Como SRCC é "maior é melhor", a perda no passo t é:
    perda(t) = max(P) - max_{a em primeiros_t}(P_a)
    """
    m = len(predicted_ranks)
    perdas = []
    
    max_p_global = np.max(true_P_row)
    
    # Ordena os índices dos algoritmos do melhor rank predito (1) para o pior (m)
    # argsort classifica do menor valor para o maior
    ordered_algs = np.argsort(predicted_ranks)
    
    for t in range(1, m + 1):
        # Seleciona os t melhores algoritmos segundo o modelo
        algoritmos_testados = ordered_algs[:t]
        
        # Maior performance real obtida por algum dos algoritmos testados
        max_p_t = np.max(true_P_row[algoritmos_testados])
        
        perda_t = max_p_global - max_p_t
        perdas.append(perda_t)
        
    auc = np.mean(perdas)
    return perdas, auc

def leave_one_dataset_out_evaluation(X, P, R, models_dict):
    """
    Executa o protocolo Leave-One-Dataset-Out para todos os modelos.
    X, P, R: DataFrames (n_datasets, n_features/n_algs)
    models_dict: Dicionário {'Nome': instancia_modelo}
    
    Retorna dicionários com SRCC, Curvas de Perda e AUC por modelo.
    """
    # Garantir alinhamento exato de linhas por dataset se forem DataFrames
    if isinstance(X, pd.DataFrame) and isinstance(P, pd.DataFrame) and isinstance(R, pd.DataFrame):
        if 'dataset' in X.columns:
            X_df = X.set_index('dataset').loc[P.index]
        elif set(X.index) == set(P.index):
            X_df = X.loc[P.index]
        else:
            X_df = X
        P_df = P.loc[P.index]
        R_df = R.loc[P.index]
        X_val, P_val, R_val = X_df.values, P_df.values, R_df.values
    else:
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        P_val = P.values if isinstance(P, pd.DataFrame) else P
        R_val = R.values if isinstance(R, pd.DataFrame) else R
        R_val = R.values if isinstance(R, pd.DataFrame) else R

    loo = LeaveOneOut()
    
    results_srcc = defaultdict(list)
    results_auc = defaultdict(list)
    results_curves = defaultdict(list)
    
    for train_index, test_index in tqdm(loo.split(X_val), total=len(X_val), desc="Leave-One-Dataset-Out"):
        X_train, X_test = X_val[train_index], X_val[test_index]
        P_train, P_test = P_val[train_index], P_val[test_index]
        R_train, R_test = R_val[train_index], R_val[test_index]
        
        test_P_row = P_test[0]
        test_R_row = R_test[0]
        
        for name, model in models_dict.items():
            # Alguns modelos (AR, MR) só precisam de R para o fit.
            # Regressores precisam de X e P ou R.
            # HARRIS precisa de X, P, R.
            if name in ['AR', 'MR']:
                model.fit(R_train)
            elif name == 'Abordagem1 (P)':
                model.fit(X_train, P_train)
            elif name == 'Abordagem2 (R)':
                model.fit(X_train, R_train)
            elif name.startswith('HARRIS'):
                model.fit(X_train, P_train, R_train)
                
            # Previsão para o dataset de teste
            pred_rank = model.predict(X_test)[0]
            
            # Métrica 1: Correlação de Spearman entre rank predito e rank real
            srcc, _ = spearmanr(pred_rank, test_R_row)
            results_srcc[name].append(srcc)
            
            # Métrica 2: Curva de Perda
            curva, auc = calculate_loss_curve(pred_rank, test_P_row)
            results_curves[name].append(curva)
            results_auc[name].append(auc)
            
    # Consolidação final (médias)
    final_results = {}
    for name in models_dict.keys():
        final_results[name] = {
            'Mean_SRCC': np.mean(results_srcc[name]),
            'All_SRCC': results_srcc[name],
            'Mean_AUC': np.mean(results_auc[name]),
            'Mean_Curve': np.mean(results_curves[name], axis=0)
        }
        
    return final_results

def plot_critical_difference_diagram(results_srcc_dict, alpha=0.05, save_path='cd_diagram.png'):
    """
    Gera o Gráfico de Diferença Crítica comparando as abordagens 
    usando o teste de Friedman seguido do pós-teste de Nemenyi.
    """
    import scikit_posthocs as sp
    import matplotlib.pyplot as plt
    
    df_results = pd.DataFrame(results_srcc_dict)
    
    # 1. Teste de Friedman (estatística não paramétrica para amostras pareadas)
    from scipy.stats import friedmanchisquare
    stat, p_val = friedmanchisquare(*[df_results[c] for c in df_results.columns])
    print(f"Teste de Friedman: p-value = {p_val:.4e}")
    
    if p_val > alpha:
        print("As abordagens não possuem diferença estatisticamente significativa global. Diagrama CD não é necessário.")
    
    # 2. Pós-teste de Nemenyi
    nemenyi_pvals = sp.posthoc_nemenyi_friedman(df_results.values)
    nemenyi_pvals.columns = df_results.columns
    nemenyi_pvals.index = df_results.columns
    
    # 3. Plot do Diagrama
    # Calcula os ranks médios para cada abordagem
    avg_ranks = df_results.rank(axis=1, ascending=False).mean()
    
    try:
        # scikit-posthocs possui a função critical_difference_diagram
        plt.figure(figsize=(10, 4), dpi=100)
        sp.critical_difference_diagram(avg_ranks, nemenyi_pvals)
        plt.title('Diagrama de Diferença Crítica (Friedman + Nemenyi)')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Diagrama de Diferença Crítica salvo em {save_path}")
    except AttributeError:
        plt.figure(figsize=(8, 6))
        sp.sign_plot(nemenyi_pvals)
        plt.title('Pós-teste de Nemenyi - Mapa de Significância')
        plt.savefig(save_path)
        plt.close()
        print(f"Heatmap de Nemenyi salvo em {save_path}")
