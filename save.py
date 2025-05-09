import os
import joblib  # Thư viện để lưu mô hình
import json

def save_model(model, model_name, fold, directory="saved_models"):
    model_dir = os.path.join(directory, model_name, f"fold_{fold}")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model.pkl")
    
    joblib.dump(model, model_path)
    

def load_model(model_name, fold, directory="best_models"):
   
    model_path = os.path.join(directory, model_name, f"fold_{fold}", "model.pkl")
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    return joblib.load(model_path)
 



def save_encoding_to_json(encoding_dict, save_path: str, target_encoding):

    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"{target_encoding}.json")

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(encoding_dict, f, ensure_ascii=False, indent=4)