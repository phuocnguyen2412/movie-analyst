import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

from settings import BASE_DIR

data_path = os.path.join(BASE_DIR, 'movies_data_processed.csv')
# Đọc dữ liệu
df = pd.read_csv(data_path)

# Tiền xử lý genres và countries
df['genres_list'] = df['genres'].apply(lambda x: [g.strip() for g in x.split(',')])
df['countries_list'] = df['countries'].apply(lambda x: [c.strip() for c in x.split(',')])

# Target encoding cho genres
genre_avg = {}
for i, genres in df['genres_list'].items():
    for g in genres:
        if g not in genre_avg:
            genre_avg[g] = []
        genre_avg[g].append(df.loc[i, 'gross'])

genre_avg = {k: np.mean(v) for k, v in genre_avg.items()}

df['genre_stat_feature'] = df['genres_list'].apply(
    lambda gl: np.mean([genre_avg.get(g, 0) for g in gl])
)

# Target encoding cho countries
country_avg = {}
for i, countries in df['countries_list'].items():
    for c in countries:
        if c not in country_avg:
            country_avg[c] = []
        country_avg[c].append(df.loc[i, 'gross'])

country_avg = {k: np.mean(v) for k, v in country_avg.items()}

df['country_stat_feature'] = df['countries_list'].apply(
    lambda cl: np.mean([country_avg.get(c, 0) for c in cl])
)

# Lựa chọn các feature cần thiết
features = ['rating', 'no_of_votes', 'budget', 'genre_stat_feature', 'country_stat_feature']
target = 'gross'

X = df[features].values
y = df[target].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)  # 0.1765 x 0.85 ≈ 0.15

