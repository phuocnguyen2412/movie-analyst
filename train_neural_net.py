from keras.src.losses import mean_absolute_error
from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from NeuralNet import Net
from data import X, y
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

y = np.log1p(y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Chuẩn hóa dữ liệu
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Chuyển sang Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Tạo DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=X_train.shape[0], shuffle=True)

    # Khởi tạo mô hình, loss và optimizer
    model = Net(X.shape[1])
    criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []

    # Huấn luyện
    for epoch in range(1000):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())


    # Đánh giá
    model.eval()
    with torch.no_grad():
        log_preds = model(X_val_tensor).numpy().flatten()
        log_true_vals = y_val_tensor.numpy().flatten()

        # Đảo ngược log1p để đưa về scale ban đầu
        preds = np.expm1(log_preds)
        true_vals = np.expm1(log_true_vals)

        mse = mean_squared_error(true_vals, preds)
        mae = mean_absolute_error(true_vals, preds)
        r2 = r2_score(true_vals, preds)

    # Vẽ train loss
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss Curve - Fold {fold + 1}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation MSE: {mse:.2f}")
    print(f"Validation R² : {r2:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss Curve - Fold {fold + 1}")
    plt.legend()
    plt.grid(True)
    plt.show()
