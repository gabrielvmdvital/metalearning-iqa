# src/features.py
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

def extract_image_features(image_path, max_dim=512):
    """
    Extrai atributos numéricos de uma única imagem (Específicas do Domínio).
    Trataremos essas estatísticas da imagem como os 'atributos' do nosso dataset.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        img_np = np.array(img)
        gray_img = rgb2gray(img_np)
        
        brilho = np.mean(gray_img)
        contraste = np.std(gray_img)
        # SNR (Signal-to-Noise Ratio): razão entre a média (sinal) e o desvio padrão (ruído)
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
    except Exception as e:
        return {k: np.nan for k in ['brilho', 'contraste', 'entropia', 'assimetria', 'curtose', 'snr', 'width', 'height']}

def build_meta_features_matrix(df_metadata):
    """
    Constrói a Matriz de Meta-características (X) mapeando para as famílias do OpenML.
    """
    datasets = df_metadata['dataset_name'].unique()
    x_matrix = []
    
    for dataset in tqdm(datasets, desc="Extraindo Meta-características"):
        df_ds = df_metadata[df_metadata['dataset_name'] == dataset]
        dist_paths = df_ds['dist_path'].unique()
        
        # 1. Gerais / de dimensão
        ds_features = {
            'dataset': dataset,
            'NumberOfInstances': len(dist_paths),
            'NumberOfFeatures': 8, # 8 atributos extraídos (inclui width e height)
            'NumberOfClasses': df_ds['dist_level'].nunique() if 'dist_level' in df_ds.columns else 0,
        }
        
        # 2. Balanceamento de classes (Mapeado para dist_level, já que cada dataset é 1 única distorção)
        if 'dist_level' in df_ds.columns and not df_ds['dist_level'].empty:
            level_counts = df_ds['dist_level'].value_counts()
            maj_size = level_counts.max()
            min_size = level_counts.min()
            ds_features['MajorityClassSize'] = maj_size
            ds_features['MinorityClassSize'] = min_size
            ds_features['MajorityClassPercentage'] = (maj_size / len(df_ds)) * 100
            ds_features['MinorityClassPercentage'] = (min_size / len(df_ds)) * 100
        else:
            ds_features['MajorityClassSize'] = 0
            ds_features['MinorityClassSize'] = 0
            ds_features['MajorityClassPercentage'] = 0.0
            ds_features['MinorityClassPercentage'] = 0.0
            
        # Coleta os atributos de todas as instâncias (imagens)
        img_feats = defaultdict(list)
        for path in dist_paths:
            feats = extract_image_features(path)
            if not np.isnan(feats['brilho']):
                for k, v in feats.items():
                    img_feats[k].append(v)
                    
        # 2. Estatísticas
        all_means = [np.mean(img_feats[k]) for k in img_feats]
        all_stds = [np.std(img_feats[k]) for k in img_feats]
        all_skews = [skew(img_feats[k]) for k in img_feats]
        all_kurts = [kurtosis(img_feats[k]) for k in img_feats]
        
        ds_features['MeanMeansOfNumericAtts'] = np.mean(all_means)
        ds_features['MeanStdDevOfNumericAtts'] = np.mean(all_stds)
        ds_features['MeanSkewnessOfNumericAtts'] = np.mean(all_skews)
        ds_features['MeanKurtosisOfNumericAtts'] = np.mean(all_kurts)
        
        # 3. Teoria da informação
        ds_features['MeanAttributeEntropy'] = np.mean(img_feats['entropia'])
        
        mean_snr = np.mean(img_feats['snr'])
        ds_features['NoiseToSignalRatio'] = 1.0 / mean_snr if mean_snr > 0 else 0.0
        
        # 4. Específicas do Domínio
        ds_features['IQA_MeanBrightness'] = np.mean(img_feats['brilho'])
        ds_features['IQA_MeanContrast'] = np.mean(img_feats['contraste'])
        
        x_matrix.append(ds_features)
        
    return pd.DataFrame(x_matrix)
