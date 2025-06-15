import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def split(input_file, percentage):
    data = pd.read_csv(input_file)

    train_data, validation_data = train_test_split(data, test_size=percentage, random_state=42, stratify=data['knight' if 'knight' in data.columns else None])

    train_data.to_csv("Training_knight.csv", index=False)
    validation_data.to_csv("Validation_knight.csv", index=False)

    print("Data has been split successfully:")
    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(validation_data)} rows")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    if input_file != "Train_knight.csv":
        print("Error: The input file must be exactly 'Train_knight.csv'")
        sys.exit(1)

    split(input_file, 0.2)