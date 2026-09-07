import torch
import pyiqa
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from scipy.stats import spearmanr
from tqdm import tqdm
from msa_iqa import MSAIQA
import torchvision.transforms as T

def load_image_tensor(path, max_dim=512):
    """Carrega uma imagem como tensor PyTorch com redimensionamento otimizado."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(img).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tensor.to(device)

def get_metrics_dict():
    """Inicializa os algoritmos de IQA via PyIQA e customizados."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Métricas PyIQA
    metrics = {
        'ssim': pyiqa.create_metric('ssim', as_loss=False).to(device),
        'psnr': pyiqa.create_metric('psnr', as_loss=False).to(device),
        'lpips': pyiqa.create_metric('lpips', as_loss=False).to(device),
        'brisque': pyiqa.create_metric('brisque', as_loss=False).to(device),
        'niqe': pyiqa.create_metric('niqe', as_loss=False).to(device),
    }
    
    # MSA-IQA (Multilayer Spatial Attention for IQA)
    meta_iqa = MSAIQA().to(device)
    meta_iqa.eval()
    metrics['msa_iqa'] = meta_iqa
    
    return metrics

def calculate_performance_matrix(df_metadata, metrics_dict):
    """
    Constrói a Matriz de Performances (P).
    Otimizado: Carrega a imagem do disco UMA ÚNICA VEZ por par e reutiliza os tensores.
    """
    results = []
    datasets = df_metadata['dataset_name'].unique()
    
    msa_transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for dataset in tqdm(datasets, desc="Processando Datasets"):
        df_ds = df_metadata[df_metadata['dataset_name'] == dataset]
        folds = df_ds['ref_id'].unique()
        
        for fold in folds:
            df_fold = df_ds[df_ds['ref_id'] == fold]
            mos_real = df_fold['mos'].values
            
            # 1. Carregar a imagem de referência (1 por fold)
            ref_path = df_fold.iloc[0]['ref_path']
            ref_tensor = None
            if ref_path != '':
                try:
                    ref_tensor = load_image_tensor(ref_path)
                except Exception:
                    pass
            
            # 2. Pré-carregar todas as imagens distorcidas do fold UMA ÚNICA VEZ
            dist_tensors = []
            for _, row in df_fold.iterrows():
                try:
                    dist_tensors.append(load_image_tensor(row['dist_path']))
                except Exception:
                    dist_tensors.append(None)
            
            # 3. Avaliar cada algoritmo reutilizando os tensores em memória
            fold_results = {'dataset': dataset, 'fold': fold}
            
            for alg_name, metric_fn in metrics_dict.items():
                scores = []
                for dist_tensor in dist_tensors:
                    if dist_tensor is None:
                        scores.append(np.nan)
                        continue
                    try:
                        with torch.no_grad():
                            if alg_name == 'msa_iqa':
                                d_tensor = msa_transform(dist_tensor)
                                score, _ = metric_fn(d_tensor)
                                score = score.item()
                            elif hasattr(metric_fn, 'lower_better') and getattr(metric_fn, 'metric_mode', 'FR') == 'FR':
                                if ref_tensor is None:
                                    score = np.nan
                                else:
                                    score = metric_fn(dist_tensor, ref_tensor).item()
                            else:
                                score = metric_fn(dist_tensor).item()
                        scores.append(score)
                    except Exception:
                        scores.append(np.nan)
                
                # Calcular SRCC entre mos_real e os scores do algoritmo
                valid_mos = []
                valid_scores = []
                for m, s in zip(mos_real, scores):
                    if not np.isnan(s):
                        valid_mos.append(m)
                        valid_scores.append(s)
                
                if len(valid_scores) > 1:
                    srcc, _ = spearmanr(valid_mos, valid_scores)
                else:
                    srcc = 0.0
                fold_results[alg_name] = srcc
                
            results.append(fold_results)
            
    df_p = pd.DataFrame(results)
    
    df_p_agg = df_p.groupby('dataset').mean(numeric_only=True).reset_index()
    return df_p_agg

