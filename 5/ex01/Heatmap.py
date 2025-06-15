import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import use
use('TkAgg')

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    if 'knight' in df.columns:
        df['knight'] = df['knight'].map({'Jedi': 0, 'Sith': 1})
    return df

def compute_correlation_matrix(df):
    return df.corr(numeric_only=True)

def plot_heatmap(corr_matrix, title="Correlation Coefficient Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="magma",
        square=True,
        cbar=True,
        linewidths=0,
        xticklabels=True,
        yticklabels=True
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    file_path = "Train_knight.csv"
    df = load_and_prepare_data(file_path)
    corr = compute_correlation_matrix(df)
    plot_heatmap(corr)

if __name__ == "__main__":
    main()

