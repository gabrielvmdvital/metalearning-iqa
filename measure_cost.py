import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.features import extract_image_features
from src.data_loader import build_master_dataframe

print("Carregando base de metadados...")
df = build_master_dataframe(Path('data/raw'))

if df.empty:
    print('Sem dados encontrados em data/raw')
    exit()

datasets = df['dataset_name'].unique()
tempos_por_dataset = []

print(f"Iniciando cálculo de custo computacional para {len(datasets)} datasets...")

for dataset in tqdm(datasets, desc="Processando Datasets"):
    df_ds = df[df['dataset_name'] == dataset]
    paths = df_ds['dist_path'].tolist()
    
    start_ds = time.time()
    for p in paths:
        extract_image_features(p)
    end_ds = time.time()
    
    total_time_ds = end_ds - start_ds
    tempos_por_dataset.append(total_time_ds)
    # Print para você poder acompanhar o andamento no log do SLURM
    print(f"Dataset '{dataset}' ({len(paths)} imagens) extraído em: {total_time_ds:.2f} segundos")

media_dataset = np.mean(tempos_por_dataset)
mediana_dataset = np.median(tempos_por_dataset)

print("\n" + "-" * 50)
print("=== RESULTADOS FINAIS DE CUSTO COMPUTACIONAL ===")
print(f"Tempo MÉDIO gasto por dataset: {media_dataset:.2f} segundos")
print(f"Tempo MEDIANO gasto por dataset: {mediana_dataset:.2f} segundos")
print(f"Tempo TOTAL para extrair as meta-features de todos os dados: {sum(tempos_por_dataset):.2f} segundos")
print("-" * 50)
