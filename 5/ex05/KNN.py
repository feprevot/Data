import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib import use
use('TkAgg')


def split_train_validation(input_file, test_size=0.2):
    data = pd.read_csv(input_file)
    train_data, val_data = train_test_split(
        data, test_size=test_size,
        stratify=data['knight'] if 'knight' in data.columns else None
    )
    train_data.to_csv("Training_knight.csv", index=False)
    val_data.to_csv("Validation_knight.csv", index=False)


def load_data(train_path, val_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print("file not found:", e)
        sys.exit(1)
    return train_df, val_df, test_df


def preprocess_data(train_df, val_df, test_df):
    X_train = train_df.drop(columns=["knight"])
    y_train = train_df["knight"]
    X_val = val_df.drop(columns=["knight"])
    y_val = val_df["knight"]
    X_test = test_df

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, scaler


def search_best_k(X_train_scaled, y_train, X_val_scaled, y_val, max_k=30):
    best_k = 1
    best_f1 = 0
    f1_scores = []

    for k in range(1, min(max_k, len(X_train_scaled))):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_val_pred = knn.predict(X_val_scaled)
        f1 = f1_score(y_val, y_val_pred, pos_label="Jedi", average="binary")
        f1_scores.append(f1)
        if f1 > best_f1:
            best_k = k
            best_f1 = f1
    return best_k, best_f1, f1_scores


def plot_f1_scores(f1_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(f1_scores)+1), f1_scores, marker='o')
    plt.title("F1-score vs Number of Neighbors (k)")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def predict_and_save(model, X_test_scaled, output_file="KNN.txt"):
    predictions = model.predict(X_test_scaled)
    with open(output_file, "w") as f:
        for label in predictions:
            f.write(label + "\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 KNN.py Train_knight.csv Test_knight.csv")
        sys.exit(1)

    train_csv = sys.argv[1]
    test_csv = sys.argv[2]

    split_train_validation(train_csv, test_size=0.2)
    train_df, val_df, test_df = load_data("Training_knight.csv", "Validation_knight.csv", test_csv)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, scaler = preprocess_data(train_df, val_df, test_df)

    best_k, best_f1, f1_scores = search_best_k(X_train_scaled, y_train, X_val_scaled, y_val)

    print(f"\nbest k : {best_k} with f1-score = {best_f1:.4f}")

    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train_scaled, y_train)

    predict_and_save(knn_model, X_test_scaled)
    plot_f1_scores(f1_scores)


if __name__ == "__main__":
    main()
