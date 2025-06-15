import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import use
use('TkAgg')

def process_file(file_name, is_train=True):

    data = pd.read_csv(file_name)
    if 'knight' not in data.columns:
        data['knight'] = 2
    elif is_train:
        data['knight'] = data['knight'].map({'Sith': 0, 'Jedi': 1})
    
    print(f"\nData Head before standardization ({file_name}):")
    print(data.head())

    scaler = StandardScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

    print(f"\nData Head after standardization ({file_name}):")
    print(data.head())

    return data

def create_scatter_plots(data, title):
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=data, x='Empowered', y='Prescience', hue='knight', palette={0: 'red', 1: 'blue', 2: 'green'})
    plt.title(title)
    plt.xlabel('Empowered')
    plt.ylabel('Prescience')
    plt.tight_layout()
    plt.show()

train_data = process_file('Train_knight.csv', is_train=True)

create_scatter_plots(train_data, 'Training Data')
