import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from save import save_encoding_to_json

from settings import BASE_DIR


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

def processing_data(df_train: pd.DataFrame, df_val: pd.DataFrame, fold: int, features, target,model_name,directory):
    # Tách các trường genres và countries
    for col in ['genres', 'countries']:
        df_train[f'{col}_list'] = _split_column(df_train, col)
        df_val[f'{col}_list'] = _split_column(df_val, col)

    # Ánh xạ encoding theo target
    genre_encoding = _compute_target_encoding(df_train['genres_list'], df_train['gross'])
    country_encoding = _compute_target_encoding(df_train['countries_list'], df_train['gross'])
    
    # Lưu encoding vào file JSON
    # save_encoding_to_json(genre_encoding, model_name=model_name, fold=fold+1, target_encoding="genre_encoded",directory=directory)
    # save_encoding_to_json(country_encoding, model_name=model_name, fold=fold+1, target_encoding="country_encoded",directory=directory)

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

    y_train = df_train[target].values
    y_val = df_val[target].values

    # Biểu đồ phân phối target
    plt.figure(figsize=(20, 10))
    for i, (df, label, color) in enumerate(zip([df_train, df_val], ['y_train', 'y_val'], ['blue', 'orange'])):
        plt.subplot(1, 2, i+1)
        sns.histplot(df["log_gross"], kde=True, color=color, label=label, bins=10)
        plt.title(f"Phân phối của {label} (thang gốc) - Fold {fold + 1}")
        plt.xlabel('Log Gross')
        plt.ylabel('Tần suất')
        plt.legend()

    return X_train, y_train, X_val, y_val

