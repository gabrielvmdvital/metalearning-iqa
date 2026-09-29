import time
import numpy as np
from pathlib import Path
from src.features import extract_image_features
from src.data_loader import build_master_dataframe

print("Carregando imagens para teste de velocidade...")
df = build_master_dataframe(Path('data/raw'))

if df.empty:
    print('Sem dados encontrados em data/raw')
    exit()
    
# Pega 100 imagens aleatórias para ter uma média justa
sample_paths = df['dist_path'].sample(100, random_state=42).tolist()

times = []
print("Extraindo features visuais de 100 imagens...")
for p in sample_paths:
    t0 = time.time()
    extract_image_features(p)
    t1 = time.time()
    times.append(t1 - t0)

mediana_img = np.median(times)
# Média de imagens por dataset no IQA é por volta de 300 imagens
tempo_dataset = mediana_img * 300

print("-" * 50)
print(f"Tempo mediano por imagem: {mediana_img:.4f} segundos")
print(f"Tempo estimado para 1 dataset (300 imagens): {tempo_dataset:.2f} segundos")
print("-" * 50)
