import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import use
use('TkAgg') 

def read_file(path):
    try:
        with open(path, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        sys.exit(1)
        
def compute_metrics(y_true, y_pred, classes):
    conf_matrix = [[0, 0], [0, 0]]
    
    total = len(y_true)

    for true, pred in zip(y_true, y_pred):
        i = classes.index(true)
        j = classes.index(pred)
        conf_matrix[i][j] += 1

    metrics = {}
    for idx, cls in enumerate(classes):
        TP = conf_matrix[idx][idx]
        FP = sum(conf_matrix[j][idx] for j in range(len(classes)) if j != idx)
        FN = sum(conf_matrix[idx][j] for j in range(len(classes)) if j != idx)
        
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        support = sum(conf_matrix[idx])

        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }

    accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / total if total != 0 else 0
    return metrics, accuracy, conf_matrix


def display_metrics(metrics, accuracy, conf_matrix, classes):
    print(f"{'':<8}{'precision':>10} {'recall':>10} {'f1-score':>10} {'total':>8}")
    for cls in classes:
        m = metrics[cls]
        print(f"{cls:<8}{m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1']:>10.2f} {m['support']:>8}")
    print(f"\n{'accuracy':<30}{accuracy:>10.2f} {'':>2} {'':>1} {sum(m['support'] for m in metrics.values()):>1}")

    sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage: ./Confusion_Matrix.py predictions.txt truth.txt")
        sys.exit(1)

    y_pred = read_file(sys.argv[1])
    y_true = read_file(sys.argv[2])

    if len(y_true) != len(y_pred):
        print("Error: Mismatched number of lines between prediction and truth files.")
        sys.exit(1)

    classes = ["Jedi", "Sith"]
    metrics, accuracy, conf_matrix = compute_metrics(y_true, y_pred, classes)
    display_metrics(metrics, accuracy, conf_matrix, classes)

if __name__ == "__main__":
    main()
