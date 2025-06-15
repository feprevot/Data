import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from matplotlib import use
from sklearn.model_selection import train_test_split
use('TkAgg')
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def split_train_validation(input_file, test_size=0.2):

    data = pd.read_csv(input_file)
    train_data, val_data = train_test_split(
        data,
        test_size=test_size,
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
    X_val_scaled = scaler.fit_transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train, y_train, X_val_scaled, y_val, X_test_scaled, scaler


def train_and_evaluate(X_train_scaled, y_train, X_val_scaled, y_val):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None,5, 10, 20,30],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2, 4],
    }

    scorer = make_scorer(f1_score, pos_label="Jedi", average="binary")

    grid = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train)


    best_model = grid.best_estimator_

    y_val_pred = best_model.predict(X_val_scaled)
    f1 = f1_score(y_val, y_val_pred, pos_label="Jedi", average="binary")
    print(f"F1-score (validation): {f1:.4f}")

    return best_model



def predict_and_save(model, X_test_scaled, output_file="Tree.txt"):
    predictions = model.predict(X_test_scaled)
    with open(output_file, "w") as f:
        for label in predictions:
            f.write(label + "\n")

def plot_random_tree(model, feature_names):
    plt.figure(figsize=(24, 12))
    plot_tree(
        model.estimators_[5],
        feature_names=feature_names,
        class_names=["Jedi", "Sith"],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("A Tree from Random Forest")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    plt.show()


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 Tree.py Train_knight.csv Test_knight.csv")
        sys.exit(1)
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]

    split_train_validation(train_csv, test_size=0.2)
    train_df, val_df, test_df = load_data("Training_knight.csv", "Validation_knight.csv", test_csv)
    X_train_df, y_train, X_val_scaled, y_val, X_test_scaled, scaler = preprocess_data(train_df, val_df, test_df)
    model = train_and_evaluate(scaler.transform(X_train_df), y_train, X_val_scaled, y_val)
    predict_and_save(model, X_test_scaled)
    plot_random_tree(model, feature_names=X_train_df.columns)

if __name__ == "__main__":
    main()
