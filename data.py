import os
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from save import save_encoding_to_json

from settings import BASE_DIR, GROSS_BIN

import pickle
import json

def _compute_target_encoding(column_lists, gross, smoothing=10):
    value_sums = {}
    value_counts = {}

    for lst, g in zip(column_lists, gross):
        for item in lst:
            value_sums[item] = value_sums.get(item, 0) + g
            value_counts[item] = value_counts.get(item, 0) + 1

    global_mean = np.mean(gross)
    encoding = {}
    for item in value_sums:
        count = value_counts[item]
        mean = value_sums[item] / count
        encoding[item] = (mean * count + global_mean * smoothing) / (count + smoothing)
    return encoding

def _split_column(df, column):
    return df[column].apply(lambda x: [item.strip() for item in x.split(',')])

def _apply_target_encoding(df, column_lists, encoding_map, new_column):
    df[new_column] = column_lists.apply(
        lambda lst: np.mean([encoding_map.get(item, 0) for item in lst])
    )

def processing_data(df_train: pd.DataFrame, df_val: pd.DataFrame, fold: int, features, target,model_name):
    save_folder = os.path.join(BASE_DIR, "best_models", f"{model_name}", f"fold_{fold+1}")
    os.makedirs(save_folder, exist_ok=True)

    # Tách các trường genres và countries
    for col in ['genres', 'countries']:
        df_train[f'{col}_list'] = _split_column(df_train, col)
        df_val[f'{col}_list'] = _split_column(df_val, col)

    # Ánh xạ encoding theo target
    genre_encoding = _compute_target_encoding(df_train['genres_list'], df_train['gross'])
    country_encoding = _compute_target_encoding(df_train['countries_list'], df_train['gross'])

    # Lưu encoding vào file JSON
    save_encoding_to_json(genre_encoding, save_path=save_folder, target_encoding="genre_encoded")
    save_encoding_to_json(country_encoding, save_path=save_folder, target_encoding="country_encoded")

    # Tạo đặc trưng thống kê từ encoding
    _apply_target_encoding(df_train, df_train['genres_list'], genre_encoding, 'genre_stat_feature')
    _apply_target_encoding(df_train, df_train['countries_list'], country_encoding, 'country_stat_feature')
    _apply_target_encoding(df_val, df_val['genres_list'], genre_encoding, 'genre_stat_feature')
    _apply_target_encoding(df_val, df_val['countries_list'], country_encoding, 'country_stat_feature')

    for col in ['country_stat_feature', 'genre_stat_feature']:
        df_val[f'log_{col}'] = np.log1p(df_val[f"{col}"])
        df_train[f'log_{col}'] = np.log1p(df_train[f"{col}"])

    # Chuẩn hóa dữ liệu
    scaler = RobustScaler()
    X_train = scaler.fit_transform(df_train[features].values)
    X_val = scaler.transform(df_val[features].values)

    # Lưu scaler
    scaler_path = os.path.join(save_folder, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to: {scaler_path}")

    y_train = df_train[target].values
    y_val = df_val[target].values

    # Biểu đồ phân phối target
    plt.figure(figsize=(20, 10))
    for i, (df, label, color) in enumerate(zip([df_train, df_val], ['y_train', 'y_val'], ['blue', 'orange'])):
        plt.subplot(1, 2, i+1)
        sns.histplot(df["log_gross"], kde=True, color=color, label=label, bins=GROSS_BIN)
        plt.title(f"Phân phối của {label} (thang gốc) - Fold {fold + 1}")
        plt.xlabel('Log Gross')
        plt.ylabel('Tần suất')
        plt.legend()

    return X_train, y_train, X_val, y_val


def load_data_test(df: pd.DataFrame, folder_path: str, fold: int, target: str, features: list[str]):
    """
    Load and preprocess test data for a specific fold.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu test.
        folder_path (str): Đường dẫn đến thư mục chứa các file encode và scaler.
        fold (int): Fold hiện tại (từ 1 đến 5).
        target (str): Tên cột target (ví dụ: 'log_gross').
        features (list): Danh sách các cột đặc trưng cần sử dụng.

    Returns:
        tuple: (X, y) - Dữ liệu đặc trưng và target sau khi xử lý.
    """
    # Tách cột genres và countries thành danh sách
    for col in ['genres', 'countries']:
        df[f'{col}_list'] = _split_column(df, col)

    # Load encoding từ file JSON
    encode_path = os.path.join(folder_path, f'fold_{str(fold)}')
    if not os.path.exists(encode_path):
        raise FileNotFoundError(f"Encoding file not found at {encode_path}")

    with open(os.path.join(encode_path, "country_encoded.json"), 'r', encoding='utf-8') as f:
        country_encoding = json.load(f)

    with open(os.path.join(encode_path,"genre_encoded.json"), 'r', encoding='utf-8') as f:
        genre_encoding = json.load(f)

    # Áp dụng target encoding cho dữ liệu test
    _apply_target_encoding(df, df['genres_list'], genre_encoding, 'genre_stat_feature')
    _apply_target_encoding(df, df['countries_list'], country_encoding, 'country_stat_feature')

    # Tạo log transform cho các đặc trưng thống kê
    for col in ['country_stat_feature', 'genre_stat_feature']:
        df[f'log_{col}'] = np.log1p(df[f"{col}"])

    # Load Scaler từ file pickle
    scaler_path = os.path.join(folder_path, f'fold_{str(fold)}', 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)

    # Chuẩn hóa dữ liệu test
    X = df[features].values
    X = scaler.transform(X)

    # Lấy giá trị target
    y = df[target].values

    return X, y