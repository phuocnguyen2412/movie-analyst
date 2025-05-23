import json
import os
import torch

from NeuralNet import Net
from settings import DEVICE

def load_models(input_dim, base_dir="best_models/neural_net"):
    models = []
    for fold in range(1, 6):
        model_dir = os.path.join(base_dir, f"fold_{fold}")
        model_path = os.path.join(model_dir, "model.pt")
        params_path = os.path.join(model_dir, "params.json")

        if not os.path.exists(model_path) or not os.path.exists(params_path):
            raise FileNotFoundError(f"Missing model or params for fold {fold}")

        # Load params
        with open(params_path, "r") as f:
             params = json.load(f)
             print(f"Loaded params for fold {fold}: {params}")


        model = Net(input_dim=input_dim,
                    num_hidden_layers=params["num_hidden_layers"],
                    dropout_rate=params["dropout_rate"]).to(DEVICE)

        # Load trọng số
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models.append(model)

    return models
