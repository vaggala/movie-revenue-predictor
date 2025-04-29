import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred, undo_log = True):
    
    if undo_log:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
   # Plot 1: Normal Scale (with scientific notation)
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red')
    plt.xlabel('Actual Revenue (in Billions $)')
    plt.ylabel('Predicted Revenue (in Billions$)')
    plt.title('Actual vs Predicted Revenue (Normal Scale)')
    plt.grid(True)
    plt.savefig('visualizations/normal_scale_plot.png', bbox_inches='tight')
    plt.show()
    
    # # Plot 2: Log Scaled
    # plt.figure(figsize=(8,6))
    # plt.scatter(y_true, y_pred, alpha=0.5)
    # plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Actual Revenue (log scale)')
    # plt.ylabel('Predicted Revenue (log scale)')
    # plt.title('Actual vs Predicted Revenue (Log-Scaled)')
    # plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    # plt.show(block=False)
    
    return {"mae": mae, "rmse": rmse, "r2": r2}