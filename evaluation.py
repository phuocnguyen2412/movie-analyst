import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error,mean_absolute_error
import pandas as pd


def _calculate_metrics(y_true, y_pred, title, calculate_real_target):
    if calculate_real_target:
        y_true = np.expm1(np.clip(y_true, 0, 20.940975))
        y_pred = np.expm1(np.clip(y_pred, 0, 20.940975))

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(title)
    print(f"R2 score: {r2:,.4f}")
    print(f"MAE: {mae:,.4f}")
    print(f"MAPE: {mape:,.2f}%")

    return r2, mae, mape

def visualize_results(y_train, train_predictions, y_test, test_predictions, model=None, df: pd.DataFrame=None, calculate_real_target=False):
    # Tính toán các chỉ số đánh giá
    train_r2, train_mae, train_mape = _calculate_metrics(
        y_train, train_predictions, title="Training Metrics", calculate_real_target=calculate_real_target
    )
    test_r2, test_mae, test_mape = _calculate_metrics(y_test, test_predictions, title="Val Metrics", calculate_real_target=calculate_real_target)

    # Vẽ biểu đồ Actual vs Predicted Values
    plt.figure(figsize=(12, 8))
    plt.scatter(y_train, train_predictions, color="blue", label="Train")
    plt.scatter(y_test, test_predictions, color="red", label="Val")
    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.show()

    if model is not None and df is not None and not df.empty:
        if hasattr(model, "feature_importances_"):
            # Nếu mô hình hỗ trợ 'feature_importances_'
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0])

            # Vẽ biểu đồ feature importance
            plt.figure(figsize=(12, 8))
            plt.barh(pos, feature_importance[sorted_idx], align="center")
            plt.yticks(pos, np.array(df.columns)[sorted_idx])
            plt.xlabel("Feature Importance")
            plt.title("Variable Importance")
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Model does not support 'feature_importances_' (e.g., HistGradientBoostingRegressor). Skipping feature importance plot.")

    return train_r2, train_mae, train_mape, test_r2, test_mae, test_mape


