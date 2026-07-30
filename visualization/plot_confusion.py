import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path,
    model_name="RF"
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(5,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()