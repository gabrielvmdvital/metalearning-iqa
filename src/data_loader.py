# src/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.io as sio

def load_tid2013(base_path):
    """
    Parser para o TID2013.
    Formato da Imagem: I{ref_id}_{dist_type}_{dist_level}.bmp
    Ex: I01_02_3.bmp -> Ref: 01, Distorção: 02, Nível: 3
    """
    base = Path(base_path)
    mos_file = base / 'mos_with_names.txt'
    
    if not mos_file.exists():
        print(f"[Aviso] TID2013 não encontrado em {base_path}")
        return pd.DataFrame()
        
    data = []
    with open(mos_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mos = float(parts[0])
                img_name = parts[1]
                
                # I01_02_3.bmp -> ref="01", dist="02", level="3"
                name_parts = img_name.split('_')
                ref_id = name_parts[0].replace('I', '')
                dist_type = name_parts[1]
                dist_level = name_parts[2].split('.')[0]
                
                ref_name = f"I{ref_id}.BMP"
                
                data.append({
                    'dataset_name': f"TID2013_Dist{dist_type}",
                    'ref_id': ref_id,
                    'ref_path': str(base / 'reference_images' / ref_name),
                    'dist_path': str(base / 'distorted_images' / img_name),
                    'mos': mos,
                    'dist_level': dist_level
                })
                
    return pd.DataFrame(data)

def load_csiq(base_path):
    """
    Parser para o CSIQ.
    Lê a planilha csiq.DMOS.xlsx e mapeia os caminhos das pastas.
    """
    base = Path(base_path)
    excel_file = base / 'csiq.DMOS.xlsx'
    
    if not excel_file.exists():
        print(f"[Aviso] CSIQ não encontrado em {base_path}")
        return pd.DataFrame()
        
    df_raw = pd.read_excel(excel_file, sheet_name='all_by_image', header=3)
    data = []
    
    for _, row in df_raw.iterrows():
        img_name = str(row['image'])
        if not img_name.endswith('.png'):
            img_name += '.png'
            
        dst_type = str(row['dst_type'])
        dst_level = row['dst_lev']
        dmos = row['dmos']
        
        img_base = img_name.replace('.png', '')
        dst_name = f"{img_base}.{dst_type}.{dst_level}.png"
        
        data.append({
            'dataset_name': f"CSIQ_{dst_type}",
            'ref_id': img_base,
            'ref_path': str(base / 'src_imgs' / img_name),
            'dist_path': str(base / 'dst_imgs' / dst_type / dst_name),
            'mos': dmos,
            'dist_level': dst_level
        })
        
    return pd.DataFrame(data)

def load_kadid10k(base_path):
    """
    Parser para KADID-10k.
    Lê o dmos.csv e mapeia as 25 distorções.
    """
    base = Path(base_path)
    csv_file = base / 'dmos.csv'
    
    if not csv_file.exists():
        print(f"[Aviso] KADID-10k não encontrado em {base_path}")
        return pd.DataFrame()
        
    df_raw = pd.read_csv(csv_file)
    data = []
    
    for _, row in df_raw.iterrows():
        dist_img = row['dist_img']
        ref_img = row['ref_img']
        dmos = row['dmos']
        
        # dist_img pattern: I01_01_01.png -> ref 01, dist 01, level 01
        parts = dist_img.split('_')
        ref_id = parts[0]
        dist_type = parts[1]
        dist_level = parts[2].split('.')[0]
        
        data.append({
            'dataset_name': f"KADID10k_Dist{dist_type}",
            'ref_id': ref_id,
            'ref_path': str(base / 'images' / ref_img),
            'dist_path': str(base / 'images' / dist_img),
            'mos': dmos,
            'dist_level': dist_level
        })
        
    return pd.DataFrame(data)

def build_full_dataframe(base_path):
    """
    Parser para LIVE Multi-distortion.
    Os metadados estão em 'Part 1/Scores.mat' e 'Part 1/Imagelists.mat'.
    Para simplificar, vamos mapear o blur_jpeg.
    """
    base = Path(base_path)
    data = []
    
    part1_scores = base / 'Part 1' / 'Scores.mat'
    part1_imgs = base / 'Part 1' / 'Imagelists.mat'
    
    if part1_scores.exists() and part1_imgs.exists():
        scores_mat = sio.loadmat(part1_scores)
        imgs_mat = sio.loadmat(part1_imgs)
        
        try:
            pass
        except Exception as e:
            print(f"[Aviso] Falha ao extrair LIVE Multi-distortion: {e}")
            
    return pd.DataFrame(data)

def load_challengedb(base_path):
    """
    Parser para LIVE In the Wild (ChallengeDB).
    Os dados estão em Data/AllMOS_release.mat e Data/AllImages_release.mat.
    """
    base = Path(base_path)
    mos_file = base / 'Data' / 'AllMOS_release.mat'
    img_file = base / 'Data' / 'AllImages_release.mat'
    
    data = []
    
    if mos_file.exists() and img_file.exists():
        mos_mat = sio.loadmat(mos_file)
        img_mat = sio.loadmat(img_file)
        
        if 'AllMOS_release' in mos_mat and 'AllImages_release' in img_mat:
            mos_array = mos_mat['AllMOS_release'][0] 
            img_array = img_mat['AllImages_release'] 
            
            for i in range(len(mos_array)):
                img_name = str(img_array[i][0][0])
                mos = float(mos_array[i])
                
                data.append({
                    'dataset_name': 'ChallengeDB',
                    'ref_id': img_name,
                    'ref_path': '', 
                    'dist_path': str(base / 'Images' / img_name),
                    'mos': mos,
                    'dist_level': '0'
                })
                
    return pd.DataFrame(data)

def build_master_dataframe(raw_data_dir):
    """
    Função orquestradora que varre os datasets e junta tudo.
    """
    raw_path = Path(raw_data_dir)
    dfs = []
    
    print("Processando TID2013...")
    df_tid = load_tid2013(raw_path / 'tid2013')
    if not df_tid.empty: dfs.append(df_tid)
        
    print("Processando CSIQ...")
    df_csiq = load_csiq(raw_path / 'csiq')
    if not df_csiq.empty: dfs.append(df_csiq)
        
    print("Processando KADID-10k...")
    df_kadid = load_kadid10k(raw_path / 'kadid10k')
    if not df_kadid.empty: dfs.append(df_kadid)
        
    print("Processando ChallengeDB (LIVE In the Wild)...")
    df_chal = load_challengedb(raw_path / 'ChallengeDB_release')
    if not df_chal.empty: dfs.append(df_chal)
        
    if dfs:
        master_df = pd.concat(dfs, ignore_index=True)
        print(f"Total de datasets extraídos (fatias de distorção): {master_df['dataset_name'].nunique()}")
        print(f"Total de pares de imagens mapeados: {len(master_df)}")
        return master_df
    else:
        print("Nenhum dado encontrado.")
        return pd.DataFrame()
