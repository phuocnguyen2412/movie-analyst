import os
import torch

from NeuralNet import Net  # Import class Net từ mô hình Neural Network của bạn
from settings import DEVICE

def load_models(input_dim, base_dir="best_models/neural_net"):
    """
    Load 5 models corresponding to 5 folds.

    Args:
        input_dim (int): Số lượng đặc trưng đầu vào của mô hình (input dimension).
        base_dir (str): Thư mục chứa các mô hình được lưu.
        device (str): Thiết bị để load mô hình ("cpu" hoặc "cuda").

    Returns:
        List[torch.nn.Module]: Danh sách 5 mô hình đã được load.
    """
    models = []
    for fold in range(1, 6):

        model_path = os.path.join(base_dir, f"fold_{fold}", "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found for fold {fold} at {model_path}")

        model = Net(input_dim=input_dim).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models.append(model)
    return models