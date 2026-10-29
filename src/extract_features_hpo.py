import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import scipy.ndimage as ndimage
import scipy.stats as stats
from tqdm import tqdm

from src.data_loader import build_master_dataframe

def extract_image_features_fast(img_path, max_dim=256):
    """
    Extrai as 9 características visuais e de Cenas Naturais (NSS) da imagem:
    1. Brilho (média de luminância)
    2. Contraste (desvio padrão de luminância)
    3. Razão Sinal-Ruído (SNR = brilho / contraste)
    4. Entropia de Shannon (complexidade da informação)
    5. Assimetria (Skewness da distribuição de cinza)
    6. Curtose (Kurtosis da distribuição de cinza)
    7. Colorfulness (Hasler & Süsstrunk, 2003)
    8. Frequência Espacial (Eskicioglu & Fisher, 1995)
    9. Extensão do Borrão (Variância do Laplaciano - Pech-Pacheco et al., 2000)
    """
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
        arr = np.array(img, dtype=float)
        # Conversão padronizada para tons de cinza (luminância ITU-R 601-2)
        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        
        b = float(np.mean(gray))
        c = float(np.std(gray))
        snr = float(b / c) if c > 1e-6 else 0.0
        
        # Entropia de Shannon discreta
        counts, _ = np.histogram(gray, bins=256, range=(0, 255))
        p = counts / (float(counts.sum()) + 1e-12)
        p = p[p > 0]
        ent = float(-np.sum(p * np.log2(p)))
        
        # Colorfulness (Hasler e Süsstrunk)
        R = arr[:, :, 0]
        G = arr[:, :, 1]
        B = arr[:, :, 2]
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        std_root = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
        mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
        color = float(std_root + 0.3 * mean_root)
        
        # Frequência Espacial
        row_diff = np.diff(gray, axis=0)
        col_diff = np.diff(gray, axis=1)
        row_f = np.mean(row_diff**2) if row_diff.size > 0 else 0.0
        col_f = np.mean(col_diff**2) if col_diff.size > 0 else 0.0
        sf = float(np.sqrt(row_f + col_f))
        
        # Variância do Laplaciano (Foco / Borramento)
        blur = float(ndimage.laplace(gray).var())
        
        flat = gray.flatten()
        sk = float(stats.skew(flat))
        kt = float(stats.kurtosis(flat))
        
        return {
            'brilho': b,
            'contraste': c,
            'snr': snr,
            'entropia': ent,
            'assimetria': sk,
            'curtose': kt,
            'colorfulness': color,
            'spatial_frequency': sf,
            'blur_extent': blur
        }
    except Exception:
        return {
            'brilho': np.nan,
            'contraste': np.nan,
            'snr': np.nan,
            'entropia': np.nan,
            'assimetria': np.nan,
            'curtose': np.nan,
            'colorfulness': np.nan,
            'spatial_frequency': np.nan,
            'blur_extent': np.nan
        }

from concurrent.futures import ThreadPoolExecutor

def _process_single_row(row):
    feat = extract_image_features_fast(row['dist_path'])
    feat['dataset_name'] = row['dataset_name']
    feat['ref_id'] = row['ref_id']
    feat['dist_path'] = row['dist_path']
    feat['mos'] = float(row['mos'])
    feat['dist_level'] = row['dist_level']
    return feat

def build_and_cache_image_features(max_workers=8):
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    output_file = processed_dir / 'image_features_dataset.csv'
    
    if output_file.exists():
        print(f"Arquivo já existe em: {output_file}")
        df = pd.read_csv(output_file)
        print(f"Carregado com {len(df)} imagens e {df['dataset_name'].nunique()} datasets.")
        return df
        
    print("Construindo master dataframe de imagens...")
    df_meta = build_master_dataframe(raw_dir)
    print(f"Total de registros a processar: {len(df_meta)}")
    
    t0 = time.time()
    rows = [row for _, row in df_meta.iterrows()]
    
    print(f"Iniciando extração multithreaded com {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(tqdm(executor.map(_process_single_row, rows), total=len(rows), desc="Extraindo features"))
        
    df_features = pd.DataFrame(records)
    
    # Imputação de NaN por mediana do próprio dataset para os poucos arquivos faltantes/corrompidos
    feature_cols = ['brilho', 'contraste', 'snr', 'entropia', 'assimetria', 'curtose', 'colorfulness', 'spatial_frequency', 'blur_extent']
    for col in feature_cols:
        df_features[col] = df_features.groupby('dataset_name')[col].transform(lambda s: s.fillna(s.median()))
        df_features[col] = df_features[col].fillna(df_features[col].median())
        
    df_features.to_csv(output_file, index=False)
    t1 = time.time()
    print(f"\nExtração concluída com sucesso em {t1 - t0:.2f} segundos!")
    print(f"Salvo em: {output_file}")
    print(f"Shape final: {df_features.shape}")
    return df_features

if __name__ == '__main__':
    build_and_cache_image_features()
