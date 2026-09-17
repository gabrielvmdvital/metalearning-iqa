import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis, spearmanr
from scipy.ndimage import laplace
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

def image_colorfulness(image):
    """Métrica de Colorfulness de Hasler e Süsstrunk."""
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    stdRoot = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
    meanRoot = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
    return stdRoot + (0.3 * meanRoot)

def spatial_frequency(image_gray):
    """Frequência espacial média."""
    row_freq = np.mean(np.diff(image_gray, axis=0)**2)
    col_freq = np.mean(np.diff(image_gray, axis=1)**2)
    return np.sqrt(row_freq + col_freq)

def blur_extent(image_gray):
    """Medida de foco/borrão usando variância do Laplaciano."""
    return laplace(image_gray).var()

def extract_image_features(image_path, max_dim=512):
    """
    Extrai atributos numéricos de uma única imagem (Específicas do Domínio).
    """
    try:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        
        img_np = np.array(img).astype(float) # for colorfulness
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
            'colorfulness': image_colorfulness(img_np),
            'spatial_frequency': spatial_frequency(gray_img),
            'blur_extent': blur_extent(gray_img),
            'width': float(w),
            'height': float(h)
        }
        return features
    except Exception as e:
        return {k: np.nan for k in ['brilho', 'contraste', 'entropia', 'assimetria', 'curtose', 'snr', 'colorfulness', 'spatial_frequency', 'blur_extent', 'width', 'height']}

def build_meta_features_matrix(df_metadata, extended=True):
    """
    Constrói a Matriz de Meta-características (X).
    Se extended=True, inclui as novas features específicas de IQA e de regressão.
    """
    datasets = df_metadata['dataset_name'].unique()
    x_matrix = []
    
    for dataset in tqdm(datasets, desc=f"Extraindo Meta-features (Extended={extended})"):
        df_ds = df_metadata[df_metadata['dataset_name'] == dataset]
        
        ds_features = {
            'dataset': dataset,
            'NumberOfInstances': len(df_ds),
            'NumberOfFeatures': 11 if extended else 8,
            'NumberOfClasses': df_ds['dist_level'].nunique() if 'dist_level' in df_ds.columns else 0,
        }
        
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
            
        img_feats = defaultdict(list)
        mos_list = []
        
        for _, row in df_ds.iterrows():
            feats = extract_image_features(row['dist_path'])
            if not np.isnan(feats['brilho']):
                for k, v in feats.items():
                    img_feats[k].append(v)
                mos_list.append(row['mos'])
                    
        base_keys = ['brilho', 'contraste', 'entropia', 'assimetria', 'curtose', 'snr', 'width', 'height']
        if not extended:
            img_feats = {k: v for k, v in img_feats.items() if k in base_keys}
            
        all_means = [np.mean(img_feats[k]) for k in img_feats]
        all_stds = [np.std(img_feats[k]) for k in img_feats]
        all_skews = [skew(img_feats[k]) for k in img_feats]
        all_kurts = [kurtosis(img_feats[k]) for k in img_feats]
        
        ds_features['MeanMeansOfNumericAtts'] = np.mean(all_means)
        ds_features['MeanStdDevOfNumericAtts'] = np.mean(all_stds)
        ds_features['MeanSkewnessOfNumericAtts'] = np.mean(all_skews)
        ds_features['MeanKurtosisOfNumericAtts'] = np.mean(all_kurts)
        
        ds_features['MeanAttributeEntropy'] = np.mean(img_feats['entropia'])
        
        mean_snr = np.mean(img_feats['snr'])
        ds_features['NoiseToSignalRatio'] = 1.0 / mean_snr if mean_snr > 0 else 0.0
        
        ds_features['IQA_MeanBrightness'] = np.mean(img_feats['brilho'])
        ds_features['IQA_MeanContrast'] = np.mean(img_feats['contraste'])
        
        if extended:
            ds_features['IQA_MeanColorfulness'] = np.mean(img_feats['colorfulness'])
            ds_features['IQA_MeanSpatialFreq'] = np.mean(img_feats['spatial_frequency'])
            ds_features['IQA_MeanBlurExtent'] = np.mean(img_feats['blur_extent'])
            
            # --- Regressão Meta-features ---
            mos_arr = np.array(mos_list)
            mos_mean = np.mean(mos_arr)
            mos_std = np.std(mos_arr)
            
            ds_features['TargetCoV'] = mos_std / mos_mean if mos_mean != 0 else 0.0
            
            corrs = []
            for k in img_feats:
                c, _ = spearmanr(img_feats[k], mos_arr)
                if not np.isnan(c):
                    corrs.append(np.abs(c))
            
            if corrs:
                ds_features['MeanCorrWithTarget'] = np.mean(corrs)
                ds_features['MaxCorrWithTarget'] = np.max(corrs)
            else:
                ds_features['MeanCorrWithTarget'] = 0.0
                ds_features['MaxCorrWithTarget'] = 0.0
                
            feat_df = pd.DataFrame(img_feats)
            corr_mat = feat_df.corr(method='spearman').abs().values
            upper_tri = corr_mat[np.triu_indices_from(corr_mat, k=1)]
            upper_tri = upper_tri[~np.isnan(upper_tri)]
            if len(upper_tri) > 0:
                ds_features['MeanFeatureCorr'] = np.mean(upper_tri)
            else:
                ds_features['MeanFeatureCorr'] = 0.0
                
            if len(mos_arr) > 2:
                X_lr = feat_df.fillna(0).values
                lr = LinearRegression()
                try:
                    lr.fit(X_lr, mos_arr)
                    preds = lr.predict(X_lr)
                    mae = np.mean(np.abs(preds - mos_arr))
                    ds_features['LinearRegressionMAE'] = mae / (mos_mean + 1e-6)
                except Exception:
                    ds_features['LinearRegressionMAE'] = 0.0
            else:
                ds_features['LinearRegressionMAE'] = 0.0
                
        x_matrix.append(ds_features)
        
    return pd.DataFrame(x_matrix)
