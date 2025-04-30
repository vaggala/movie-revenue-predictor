import matplotlib.pyplot as plt
import numpy as np


def plot_actual_vs_predicted(y_true, y_pred, undo_log=True, save_path=None):
    if undo_log:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red')
    plt.xlabel('Actual Revenue (in Billions $)')
    plt.ylabel('Predicted Revenue (in Billions$)')
    plt.title('Actual vs Predicted Revenue (Normal Scale)')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    plt.show()
    
def plot_feature_importance(model, feature_names, save_path=None):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    plt.title('Feature Importances')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    plt.show()
