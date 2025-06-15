import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib import use
use('TkAgg')

def load_and_scale_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["knight"], errors="ignore")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(X_scaled, columns=df.columns)
    return df_scaled

def calculate_vif(df_input):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_input.columns
    vif_data["VIF"] = [variance_inflation_factor(df_input.values, i) for i in range(df_input.shape[1])]
    vif_data["Tolerance"] = 1 / vif_data["VIF"]
    return vif_data

def eliminate_multicollinearity(df_scaled, max_vif=5.0):
    features = df_scaled.columns.tolist()
    iteration = 0

    while True:
        vif_df = calculate_vif(df_scaled[features])
        max_vif_value = vif_df["VIF"].max()
        if max_vif_value <= max_vif:
            break
        feature_to_remove = vif_df.sort_values("VIF", ascending=False).iloc[0]["Feature"]
        print(f"[it {iteration}] Deleted : {feature_to_remove} (VIF = {max_vif_value:.2f})")
        features.remove(feature_to_remove)
        iteration += 1

    final_vif_df = calculate_vif(df_scaled[features])
    return final_vif_df

def main():
    file_path = "Train_knight.csv"
    df_scaled = load_and_scale_data(file_path)
    final_vif_df = eliminate_multicollinearity(df_scaled, max_vif=5.0)

    print(final_vif_df)

if __name__ == "__main__":
    main()
