import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """Plot the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_roc_curve(y_test, y_score, classes):
    """Plot ROC curve for multi-class classification."""
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.title(f"Top {top_n} Feature Importances")
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("This model doesn't have feature_importances_ attribute.")


def visualize_model_performance(model, X_test, y_test, feature_names, classes):
    """Visualize various aspects of model performance."""

    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes, title=f'Confusion Matrix - {type(model).__name__}')

    # ROC Curve (for models that support predict_proba)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)
        plot_roc_curve(y_test, y_score, classes)
    else:
        print("This model doesn't support probability predictions for ROC curve.")

    # Feature Importance (for tree-based models)
    plot_feature_importance(model, feature_names)


# Usage in main function
def main():
    # ... (previous code for data loading and model training)

    for name, model in models.items():
        print(f"\nVisualizing performance for {name}...")
        visualize_model_performance(model, X_test, y_test, feature_names, classes)


if __name__ == "__main__":
    main()