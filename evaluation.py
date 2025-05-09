import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

def _calculate_metrics(y_true, y_pred):
    """
    Calculate R2, MAE, and MAPE metrics.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return r2, mae, mape

def visualize_results(df: pd.DataFrame, y_true, y_pred, bin_column='log_gross_bin', calculate_real_target=True, dataset_label="Dataset", axes=None):
    """
    Visualize results and calculate metrics on a single DataFrame for a specific dataset (train, val, or test).
    Args:
        df (pd.DataFrame): DataFrame chứa các thông tin, bao gồm cột bin.
        y_true (np.array): True target values.
        y_pred (np.array): Predicted target values.
        bin_column (str): Cột đại diện cho các bin (ví dụ: 'log_gross_bin').
        calculate_real_target (bool): Nếu True, áp dụng phép biến đổi log về giá trị thực.
        dataset_label (str): Nhãn để hiển thị tên tập dữ liệu (train, val, test).
        axes (matplotlib.axes.Axes): Nếu được truyền vào, sẽ vẽ đồ thị trên axes này.

    Returns:
        pd.DataFrame: DataFrame chứa các metric trên từng bin.
    """
    # Apply transformation if needed
    if calculate_real_target:
        y_true = np.expm1(np.clip(y_true, 0, 20.732282))
        y_pred = np.expm1(np.clip(y_pred, 0, 20.732282))

    # Add y_true and y_pred into the DataFrame
    df = df.copy()
    df['y_true'] = y_true
    df['y_pred'] = y_pred

    # Calculate metrics for each bin
    bin_results = []
    for bin_value, group in df.groupby(bin_column):
        r2, mae, mape = _calculate_metrics(group['y_true'], group['y_pred'])
        bin_results.append({
            'Bin': bin_value,
            'R2': r2,
            'MAE': mae,
            'MAPE': mape * 100,
            'Count': len(group)
        })


    # Convert results to DataFrame
    bin_metrics_df = pd.DataFrame(bin_results)

    # Calculate overall metrics
    overall_r2, overall_mae, overall_mape = _calculate_metrics(y_true, y_pred)
    print(f"Overall Metrics ({dataset_label}):")
    print(f"  R2: {overall_r2:.4f}")
    print(f"  MAE: {overall_mae:.4f}")
    print(f"  MAPE: {overall_mape * 100:.2f}%")

    # Plot metrics as bar charts (use axes if provided)
    if axes is not None:
        # Metrics to plot
        metrics = ['R2', 'MAE', 'MAPE']
        colors = ['skyblue', 'lightgreen', 'salmon']

        for i, metric in enumerate(metrics):
            axes[i].bar(
                bin_metrics_df['Bin'],
                bin_metrics_df[metric],
                color=colors[i],
                label=f"{dataset_label} {metric}"
            )
            axes[i].set_title(f"{metric} by Bin ({dataset_label})")
            axes[i].set_xlabel('Bins')
            axes[i].set_ylabel(metric)
            axes[i].legend()
    else:
        # Plot as individual figures if no axes are provided
        for metric in ['R2', 'MAE', 'MAPE']:
            plt.figure(figsize=(10, 6))
            plt.bar(
                bin_metrics_df['Bin'],
                bin_metrics_df[metric],
                color='skyblue',
                label=f"{dataset_label} {metric}"
            )
            plt.title(f"{metric} by Bin ({dataset_label})")
            plt.xlabel('Bins')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Return the metrics DataFrame
    return bin_metrics_df, overall_r2, overall_mae, overall_mape