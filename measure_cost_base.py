import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis

from src.data_loader import build_master_dataframe

def extract_image_features_base(image_path, max_dim=512):
    """
    Extrator de características puramente BASE.
    (Pula o Laplaciano, Frequência Espacial e Cor para ser o mais rápido possível)
    """
    try:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        
        img_np = np.array(img).astype(float)
        gray_img = rgb2gray(img_np)
        
        brilho = np.mean(gray_img)
        contraste = np.std(gray_img)
        snr = brilho / contraste if contraste > 0 else 0.0
        
        features = {
            'brilho': brilho,
            'contraste': contraste,
            'entropia': shannon_entropy(gray_img),
            'assimetria': skew(gray_img.flatten()),
            'curtose': kurtosis(gray_img.flatten()),
            'snr': snr,
            'width': float(w),
            'height': float(h)
        }
        return features
    except Exception:
        pass


print("Carregando base de metadados para MATRIZ BASE...")
df = build_master_dataframe(Path('data/raw'))

if df.empty:
    print('Sem dados encontrados em data/raw')
    exit()

datasets = df['dataset_name'].unique()
tempos_por_dataset = []

print(f"Iniciando cálculo de custo computacional para {len(datasets)} datasets (SOMENTE BASE)...")

for dataset in tqdm(datasets, desc="Processando Datasets"):
    df_ds = df[df['dataset_name'] == dataset]
    paths = df_ds['dist_path'].tolist()
    
    start_ds = time.time()
    for p in paths:
        extract_image_features_base(p)
    end_ds = time.time()
    
    total_time_ds = end_ds - start_ds
    tempos_por_dataset.append(total_time_ds)
    print(f"Dataset '{dataset}' ({len(paths)} imagens) - SOMENTE BASE extraído em: {total_time_ds:.2f} segundos")

media_dataset = np.mean(tempos_por_dataset)
mediana_dataset = np.median(tempos_por_dataset)

print("\n" + "-" * 50)
print("=== RESULTADOS DE CUSTO: MATRIZ BASE (S/ Visão Computacional) ===")
print(f"Tempo MÉDIO gasto por dataset: {media_dataset:.2f} segundos")
print(f"Tempo MEDIANO gasto por dataset: {mediana_dataset:.2f} segundos")
print(f"Tempo TOTAL para extrair as meta-features de todos os dados: {sum(tempos_por_dataset):.2f} segundos")
print("-" * 50)
