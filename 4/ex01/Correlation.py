import pandas as pd

csv_file = "Train_knight.csv"
data = pd.read_csv(csv_file)
data['knight'] = data['knight'].map({'Sith': 0, 'Jedi': 1})
correlation_matrix = data.corr()["knight"].sort_values(ascending=False)
print(correlation_matrix)
