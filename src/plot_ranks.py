import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_rank_distribution():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / 'data' / 'processed' / 'R_matrix.csv'
    out_path = base_dir / 'notebooks' / 'rank_distribution.png'
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Drop dataset column to keep only algorithms
    df_ranks = df.drop(columns=['dataset'])
    
    # Round ranks to nearest integer for visualization purposes 
    # (since average rank generates things like 1.5, 3.5)
    df_ranks_rounded = df_ranks.round().astype(int)
    
    # Create a frequency matrix: rows = Ranks (1 to 6), cols = Algorithms
    freq_matrix = pd.DataFrame(0, index=range(1, 7), columns=df_ranks.columns)
    
    for col in df_ranks_rounded.columns:
        counts = df_ranks_rounded[col].value_counts()
        for rank, count in counts.items():
            if rank in freq_matrix.index:
                freq_matrix.at[rank, col] = count
                
    # Plot Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(freq_matrix, annot=True, cmap='YlGnBu', fmt='g', linewidths=.5)
    plt.title('Distribuição de Ranks por Algoritmo')
    plt.xlabel('Algoritmos de IQA')
    plt.ylabel('Posição no Ranking (1 = Melhor)')
    plt.tight_layout()
    
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == '__main__':
    plot_rank_distribution()
