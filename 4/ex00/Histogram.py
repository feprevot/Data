import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use

use('TkAgg')
csv_file = "Test_knight.csv"
data = pd.read_csv(csv_file)

columns = data.columns

num_columns = len(columns)
cols_per_row = 5
num_rows = (num_columns + cols_per_row - 1) // cols_per_row

fig, axes = plt.subplots(nrows=num_rows, ncols=cols_per_row, figsize=(15, 4 * num_rows))
axes = axes.flatten()

for i, column in enumerate(columns):
    axes[i].hist(data[column], bins=30, color='blue', alpha=0.7)
    axes[i].set_xlabel(column, fontsize=8)
    axes[i].grid(axis='y', alpha=0.75)

for j in range(len(columns), len(axes)):
    fig.delaxes(axes[j])


plt.tight_layout(pad=5.0)

plt.show()

#--------------------------------------#

csv_file = "Train_knight.csv"
data = pd.read_csv(csv_file)

columns = data.columns[:-1]
labels = data.iloc[:, -1]

num_columns = len(columns)
cols_per_row = 5
num_rows = (num_columns + cols_per_row - 1) // cols_per_row

fig, axes = plt.subplots(nrows=num_rows, ncols=cols_per_row, figsize=(15, 4 * num_rows))
axes = axes.flatten()

for i, column in enumerate(columns):
    sith_data = data[column][labels == 'Sith']
    jedi_data = data[column][labels == 'Jedi']
    
    axes[i].hist(sith_data, bins=35, color='red', alpha=0.5, label='Sith')
    axes[i].hist(jedi_data, bins=35, color='blue', alpha=0.5, label='Jedi')
    axes[i].set_xlabel(column, fontsize=8)
    axes[i].grid(axis='y', alpha=0.75)
    axes[i].legend(fontsize=8)

for j in range(len(columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=5.0)

plt.show()
