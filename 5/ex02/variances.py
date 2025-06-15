import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import use
use('TkAgg')

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["knight"], errors="ignore")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled, df.columns

def perform_pca(X_scaled):
    pca = PCA()
    pca.fit(X_scaled)
    explained_var = pca.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)
    return explained_var, cumulative_var

def plot_variance_curve(cumulative_var, num_components_90):
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_var, marker='o')
    plt.axhline(y=90, color='r', linestyle='--', label='90% variance')
    plt.axvline(x=num_components_90 - 1, color='g', linestyle='--', label=f'{num_components_90} components')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    file_path = "Train_knight.csv"
    X_scaled, feature_names = load_and_preprocess_data(file_path)
    explained_var, cumulative_var = perform_pca(X_scaled)

    print("Variances (Percentage):")
    print(explained_var)
    print("\nCumulative Variances (Percentage):")
    print(cumulative_var)

    num_components_90 = np.argmax(cumulative_var >= 90) + 1
    print(f"\nNumber of components to reach 90% variance: {num_components_90}")

    plot_variance_curve(cumulative_var, num_components_90)

if __name__ == "__main__":
    main()
