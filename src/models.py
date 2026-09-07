import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from scipy.stats import spearmanr

class BaselineRanker(BaseEstimator):
    """Implementa abordagens de rank fixo: AR (Rank Médio) e MR (Rank Mediano)."""
    def __init__(self, method='AR'):
        self.method = method
        self.fixed_rank = None
        
    def fit(self, R_matrix):
        """
        R_matrix: array ou DataFrame de formato (n_datasets, n_algoritmos).
        """
        if isinstance(R_matrix, pd.DataFrame):
            R_matrix = R_matrix.values
            
        if self.method == 'AR':
            agg_rank = np.mean(R_matrix, axis=0)
        elif self.method == 'MR':
            agg_rank = np.median(R_matrix, axis=0)
        else:
            raise ValueError("Method must be 'AR' or 'MR'")
        
        self.fixed_rank = rankdata(agg_rank, method='average')
        return self
        
    def predict(self, X):
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.tile(self.fixed_rank, (n_samples, 1))

class SignificantWinsRanker(BaseEstimator):
    """Implementa a abordagem de Vitórias Significativas baseada em teste de Wilcoxon."""
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.fixed_rank = None
        
    def fit(self, P_folds_df, alg_cols):
        """
        Calcula o ranking global baseado em vitórias significativas.
        Para cada par de algoritmos e para CADA DATASET, faz o teste de Wilcoxon.
        """
        n_algs = len(alg_cols)
        wins = np.zeros(n_algs)
        
        datasets = P_folds_df['dataset'].unique()
        
        for ds in datasets:
            df_ds = P_folds_df[P_folds_df['dataset'] == ds]
            
            for i in range(n_algs):
                for j in range(n_algs):
                    if i == j: continue
                    
                    alg1_scores = df_ds[alg_cols[i]].values
                    alg2_scores = df_ds[alg_cols[j]].values
                    
                    diff = alg1_scores - alg2_scores
                    if np.all(diff == 0):
                        continue
                    
                    try:
                        stat, p_val = wilcoxon(alg1_scores, alg2_scores)
                        if p_val < self.alpha:
                            if np.mean(alg1_scores) > np.mean(alg2_scores):
                                wins[i] += 1
                    except ValueError:
                        pass
        
        self.fixed_rank = rankdata(-wins, method='average')
        return self
        
    def predict(self, X):
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.tile(self.fixed_rank, (n_samples, 1))


class MetaRegressor(BaseEstimator):
    """Regressão Múltipla com Random Forest. Prediz P ou R."""
    def __init__(self, target='R', n_estimators=100, random_state=42):
        self.target = target
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        preds = self.model.predict(X)
        
        if self.target == 'R':
            return np.array([rankdata(p, method='average') for p in preds])
        elif self.target == 'P':
            return np.array([rankdata(-p, method='average') for p in preds])


class HarrisNode:
    def __init__(self, depth=0, max_depth=5, lambd=0.5):
        self.depth = depth
        self.max_depth = max_depth
        self.lambd = lambd
        
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None
        
        self.prediction_R = None
        
    def calculate_loss(self, P, R):
        if len(P) == 0: return 0.0
        
        mean_P = np.mean(P, axis=0)
        loss_reg = np.mean(np.sum((P - mean_P)**2, axis=1))
        
        mean_R = np.mean(R, axis=0)
        loss_rank = np.mean(np.sum((R - mean_R)**2, axis=1))
        
        return self.lambd * loss_rank + (1 - self.lambd) * loss_reg
        
    def fit(self, X, P, R):
        n_samples, n_features = X.shape
        
        self.prediction_R = np.mean(R, axis=0)
        
        if self.depth >= self.max_depth or n_samples < 2:
            return
            
        best_loss = float('inf')
        best_split = None
        
        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = X[:, feat] > thresh
                
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                    
                loss_left = self.calculate_loss(P[left_mask], R[left_mask])
                loss_right = self.calculate_loss(P[right_mask], R[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                total_loss = (n_left * loss_left + n_right * loss_right) / n_samples
                
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_split = (feat, thresh, left_mask, right_mask)
                    
        if best_split is not None:
            self.split_feature, self.split_threshold, left_mask, right_mask = best_split
            
            self.left = HarrisNode(depth=self.depth+1, max_depth=self.max_depth, lambd=self.lambd)
            self.left.fit(X[left_mask], P[left_mask], R[left_mask])
            
            self.right = HarrisNode(depth=self.depth+1, max_depth=self.max_depth, lambd=self.lambd)
            self.right.fit(X[right_mask], P[right_mask], R[right_mask])

    def predict_one(self, x):
        if self.left is None or self.right is None:
            return self.prediction_R
        
        if x[self.split_feature] <= self.split_threshold:
            return self.left.predict_one(x)
        else:
            return self.right.predict_one(x)


class HarrisForest(BaseEstimator):
    """
    Floresta híbrida (HARRIS) com divisão baseada em uma combinação 
    convexa de perdas (L = λL_rank + (1-λ)L_regressão).
    Versão implementada em Python puro para o projeto.
    """
    def __init__(self, lambd=0.5, n_estimators=10, max_depth=5):
        self.lambd = lambd
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, P, R):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(P, pd.DataFrame): P = P.values
        if isinstance(R, pd.DataFrame): R = R.values
            
        n_samples = X.shape[0]
        self.trees = []
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            tree = HarrisNode(max_depth=self.max_depth, lambd=self.lambd)
            tree.fit(X[indices], P[indices], R[indices])
            self.trees.append(tree)
            
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
            
        preds = []
        for x in X:
            tree_preds = [tree.predict_one(x) for tree in self.trees]
            avg_pred = np.mean(tree_preds, axis=0)
            preds.append(rankdata(avg_pred, method='average'))
            
        return np.array(preds)
