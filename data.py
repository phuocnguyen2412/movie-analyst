import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

def processing_data(df_train: pd.DataFrame, df_val: pd.DataFrame, fold: int):
    df_train['genres_list'] = df_train['genres'].apply(lambda x: [g.strip() for g in x.split(',')])
    df_train['countries_list'] = df_train['countries'].apply(lambda x: [c.strip() for c in x.split(',')])
    df_val['genres_list'] = df_val['genres'].apply(lambda x: [g.strip() for g in x.split(',')])
    df_val['countries_list'] = df_val['countries'].apply(lambda x: [c.strip() for c in x.split(',')])

    genre_encoding = _compute_target_encoding(df_train['genres_list'], df_train['gross'])
    country_encoding = _compute_target_encoding(df_train['countries_list'], df_train['gross'])

    df_train['genre_stat_feature'] = df_train['genres_list'].apply(
        lambda gl: np.mean([genre_encoding.get(g, 0) for g in gl])
    )
    df_train['country_stat_feature'] = df_train['countries_list'].apply(
        lambda cl: np.mean([country_encoding.get(c, 0) for c in cl])
    )

    df_val['genre_stat_feature'] = df_val['genres_list'].apply(
        lambda gl: np.mean([genre_encoding.get(g, 0) for g in gl])
    )
    df_val['country_stat_feature'] = df_val['countries_list'].apply(
        lambda cl: np.mean([country_encoding.get(c, 0) for c in cl])
    )

    #features = ['meta_score', 'rating', 'no_of_votes', 'budget', 'genre_stat_feature', 'country_stat_feature', "no_of_votes", 'release_date']
    features = ['rating', 'no_of_votes', 'budget', 'genre_stat_feature', 'country_stat_feature', "no_of_votes"]

    target = 'log_gross'

    X_train = df_train[features].values
    y_train = df_train[target].values

    X_val = df_val[features].values
    y_val = df_val[target].values

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)



    # Phân phối của y_train
    plt.subplot(1, 2, 1)
    sns.histplot(df_train["gross"], kde=True, color='blue', label='y_train', bins=10)
    plt.title(f"Phân phối của y_train (thang gốc) - Fold {fold + 1}")
    plt.xlabel('Gross')
    plt.ylabel('Tần suất')
    plt.legend()

    # Phân phối của y_val
    plt.subplot(1, 2, 2)
    sns.histplot(df_val["gross"], kde=True, color='orange', label='y_val', bins=10)
    plt.title(f"Phân phối của y_val (thang gốc) - Fold {fold + 1}")
    plt.xlabel('Gross')
    plt.ylabel('Tần suất')
    plt.legend()

    return X_train, y_train, X_val, y_val