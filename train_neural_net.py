import os

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from NeuralNet import Net

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from data import processing_data
from evaluation import visualize_results
from settings import BASE_DIR, DEVICE
import pandas as pd


data_path = os.path.join(BASE_DIR,"dataset", 'movies_data_processed_v2.csv')

df = pd.read_csv(data_path)
df['log_gross'] = np.log1p(df['gross'])
df['log_gross_bin'] = pd.qcut(df['log_gross'], q=10, labels=False)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()

PATIENCE = 150
BEST_MODEL_DIR = os.path.join(BASE_DIR, "best_models")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


all_fold_r2 = []
all_fold_mape = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['log_gross_bin'])):
    print(f"Fold {fold + 1}")

    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    X_train, y_train, X_val, y_val = processing_data(df_train, df_val, fold=fold)


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Tạo DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=X_train.shape[0], shuffle=True)


    plt.tight_layout()
    plt.show()

    # Khởi tạo mô hình, loss và optimizer
    model = Net(X_train.shape[1]).to(DEVICE)
    criterion = nn.SmoothL1Loss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(BEST_MODEL_DIR, f"best_model_fold_{fold + 1}.pt")

    # Huấn luyện
    for epoch in range(2000):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())


        # Đánh giá
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor.to(DEVICE))
            val_loss = criterion(val_preds, y_val_tensor.to(DEVICE)).item()
            val_losses.append(val_loss)

            # Chuyển sang numpy để tính các metric trên scale gốc
            log_preds = val_preds.cpu().numpy().flatten()
            log_true_vals = y_val_tensor.cpu().numpy().flatten()

            preds = np.expm1(log_preds)
            true_vals = np.expm1(log_true_vals)

            mse = mean_squared_error(true_vals, preds)
            mae = mean_absolute_error(true_vals, preds)
            r2 = r2_score(true_vals, preds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Vẽ train và val loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss (SmoothL1)", alpha=0.7)
    plt.plot(range(0, len(train_losses), len(train_losses) // len(val_losses)), val_losses,
             label="Validation Loss (SmoothL1)", color='orange', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - Fold {fold + 1}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    model.eval()
    with torch.no_grad():
        test_predictions = model(X_val_tensor.to(DEVICE))
        test_predictions = test_predictions.cpu().numpy().flatten()
        y_test = y_val_tensor.cpu().numpy().flatten()

        train_predictions = model(X_train_tensor.to(DEVICE))
        train_predictions = train_predictions.cpu().numpy().flatten()
        y_train = y_train_tensor.cpu().numpy().flatten()


        _, _, _, _, test_r2, test_mse, test_msle, test_mape = visualize_results(y_train, train_predictions, y_test, test_predictions, is_logged=True)

        all_fold_r2.append(test_r2)
        all_fold_mape.append(test_mape)


avg_r2 = np.mean(all_fold_r2)
avg_mape = np.mean(all_fold_mape)

print("\n===== Cross-Validation Average Results =====")
print(f"Avg R²: {avg_r2:.4f} (±{np.std(all_fold_r2):.4f})")
print(f"Avg MAPE Loss: {avg_mape:.4f} (±{np.std(avg_mape):.4f})")

# Optionally, plot average metrics across folds
plt.figure(figsize=(12, 8))
fold_indices = list(range(1, len(all_fold_mape) + 1))

plt.subplot(2, 2, 1)
plt.bar(fold_indices, all_fold_r2)
plt.axhline(y=avg_r2, color='r', linestyle='-', label=f'Average: {avg_r2:.2f}')
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.title('MSE by Fold')
plt.legend()

plt.subplot(2, 2, 2)
plt.bar(fold_indices, all_fold_mape)
plt.axhline(y=avg_mape, color='r', linestyle='-', label=f'Average: {avg_mape:.2f}')
plt.xlabel('Fold')
plt.ylabel('MAE')
plt.title('MAE by Fold')
plt.legend()



plt.tight_layout()
plt.show()


