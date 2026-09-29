import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

print("Carregando matrizes...")
data_dir = Path('data/processed')
notebooks_dir = Path('notebooks')

if not (data_dir / 'X_matrix_estendida.csv').exists():
    print("Erro: X_matrix_estendida.csv não encontrado. Rode o main.py primeiro.")
    exit()

df_X = pd.read_csv(data_dir / 'X_matrix_estendida.csv')
if 'dataset' in df_X.columns:
    df_X = df_X.drop('dataset', axis=1)
df_R = pd.read_csv(data_dir / 'R_matrix.csv', index_col=0)

# Treinando um RandomForestRegressor para prever os ranks
print("Treinando MetaRegressor para calcular importâncias...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(df_X.values, df_R.values)

importances = model.feature_importances_
feature_names = df_X.columns

# Ordenando importâncias
indices = np.argsort(importances)

plt.figure(figsize=(12, 10))
plt.title('Importância das Meta-características (Abordagem de Regressão em Ranks)', fontsize=14)
plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
plt.xlabel('Importância Relativa (Gini Importance)', fontsize=12)

# Adicionando os valores numéricos no gráfico
for i, v in enumerate(importances[indices]):
    plt.text(v, i, f' {v:.4f}', va='center', fontsize=9)

plt.tight_layout()
output_path = notebooks_dir / 'feature_importance.png'
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Gráfico de importância gerado com sucesso!")
print(f"Salvo em: {output_path}")

# Imprimindo o Top 5 no console
print("\n=== TOP 5 META-CARACTERÍSTICAS ===")
top5_idx = indices[-5:][::-1]
for i in top5_idx:
    print(f"{feature_names[i]}: {importances[i]:.4f}")
