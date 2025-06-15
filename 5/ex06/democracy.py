import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def split_train_validation(input_file, test_size=0.2):
    data = pd.read_csv(input_file)
    train_data, val_data = train_test_split(
        data, test_size=test_size,
        stratify=data['knight'] if 'knight' in data.columns else None
    )
    train_data.to_csv("Training_knight.csv", index=False)
    val_data.to_csv("Validation_knight.csv", index=False)

def load_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
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

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled

def search_best_k(X_train_scaled, y_train, X_val_scaled, y_val, max_k=30):
    best_k = 1
    best_f1 = 0

    for k in range(1, min(max_k, len(X_train_scaled))):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_val_pred = knn.predict(X_val_scaled)
        f1 = f1_score(y_val, y_val_pred, pos_label="Jedi", average="binary")
        if f1 > best_f1:
            best_k = k
            best_f1 = f1
    return best_k

def train_best_tree(X_train_scaled, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    }

    scorer = make_scorer(f1_score, pos_label="Jedi", average="binary")

    grid = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        scoring=scorer,

    )
    grid.fit(X_train_scaled, y_train)
    return grid.best_estimator_

def train_voting_classifier(X_train_scaled, y_train, X_val_scaled, y_val):
    best_k = search_best_k(X_train_scaled, y_train, X_val_scaled, y_val)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    rf = train_best_tree(X_train_scaled, y_train)
    lr = LogisticRegression(max_iter=1000)

    voting = VotingClassifier(estimators=[
        ("knn", knn),
        ("rf", rf),
        ("lr", lr)
    ], voting="hard")

    voting.fit(X_train_scaled, y_train)
    y_val_pred = voting.predict(X_val_scaled)
    f1 = f1_score(y_val, y_val_pred, pos_label="Jedi", average="binary")
    print(f"F1-score (Voting Classifier): {f1:.4f}")

    return voting

def predict_and_save(model, X_test_scaled, output_file="Voting.txt"):
    predictions = model.predict(X_test_scaled)
    with open(output_file, "w") as f:
        for label in predictions:
            f.write(label + "\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 Democracy.py Train_knight.csv Test_knight.csv")
        sys.exit(1)

    train_csv = sys.argv[1]
    test_csv = sys.argv[2]

    split_train_validation(train_csv)
    train_df, val_df, test_df = load_data("Training_knight.csv", "Validation_knight.csv", test_csv)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled = preprocess_data(train_df, val_df, test_df)

    model = train_voting_classifier(X_train_scaled, y_train, X_val_scaled, y_val)
    predict_and_save(model, X_test_scaled)

if __name__ == "__main__":
    main()