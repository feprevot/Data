

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import use
use('TkAgg')
import seaborn as sns
import numpy as np

train_data = pd.read_csv('Train_knight.csv')
test_data = pd.read_csv('Test_knight.csv')

train_data['knight'] = train_data['knight'].map({'Sith': 0, 'Jedi': 1})
test_data['knight'] = 2

sns.set(style="whitegrid")

def create_scatter_plots(data, title):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(data=data, x='Empowered', y='Prescience', hue='knight', palette={0: 'red', 1: 'blue', 2: 'green'})
    plt.title(f"{title} - Separated Clusters")
    plt.xlabel('Empowered')
    plt.ylabel('Prescience')

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=data, x='Survival', y='Deflection', hue='knight', palette={0: 'red', 1: 'blue', 2: 'green'})
    plt.title(f"{title} - Mixed Clusters")
    plt.xlabel('Survival')
    plt.ylabel('Deflection')

    plt.tight_layout()
    plt.show()

create_scatter_plots(train_data, 'Training Data')
create_scatter_plots(test_data, 'Training Data')
