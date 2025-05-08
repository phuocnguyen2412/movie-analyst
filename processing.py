import pandas as pd


import numpy as np
import re

def convert_votes(vote):
    # Chuyển giá trị votes từ string có đơn vị M, K sang int
    try:
        vote = str(vote)
        if vote is None or vote == '':
            return np.nan
        elif vote.endswith("K"):
            return int(float(vote[:-1]) * 1000)
        elif vote.endswith("M"):
            return int(float(vote[:-1]) * 1000000)
        return int(vote)
    except:
        return np.nan

def convert_released_day(date):
    #Chỉ lấy năm
    try:
        if pd.isna(date):
            return None
        # Tìm tất cả số có 4 chữ số
        years = re.findall(r"\b(\d{4})\b", date)
        if years:
            return int(years[-1])  # lấy năm cuối cùng nếu có nhiều năm
        return None
    except:
        return np.nan

def convert_gross_budget(money):
    # Chuyển giá trị tiền từ string có đơn vị $, , sang float
    try:
        money = str(money).split()[0]
        if money is None or money == '':
            return np.nan
        if money.startswith("$"):
            money = money[1:]
        money = money.replace(",", "")
        return float(money)
    except:
        return np.nan

def make_success_label(row):
    # Tạo nhãn kết quả cho dữ liệu dựa trên các cột rating, ROI, no_of_votes, meta_score
    rating = row["rating"]
    ROI = row["ROI"]
    no_of_votes = row["no_of_votes"]
    meta_score = row["meta_score"]
    # Kiểm tra tất cả đều NaN
    try:
        if pd.isna(rating) and pd.isna(ROI) and pd.isna(no_of_votes) and pd.isna(meta_score):
            return "Unknown"

        if ROI > 1 or rating > 7.0 or no_of_votes > 100000 or meta_score > 70:
            return "Success"
        return "Fail"
    except:
        return "Unknown"

movie_df = pd.read_csv('dataset/raw_data.csv')

# Xóa các phim trùng lặp
movie_df = movie_df.drop_duplicates(subset=["name"], keep="first")

#drop những cột không cần thiết
movie_df.drop(columns=["url","type", "name"], inplace=True)

movie_df["gross"] = movie_df["gross"].apply(convert_gross_budget).astype('float')

movie_df["budget"]= movie_df["budget"].apply(convert_gross_budget).astype('float')

movie_df["release_date"] = movie_df["release_date"].apply(convert_released_day).astype("str")

movie_df["no_of_votes"] = movie_df["no_of_votes"].apply(convert_votes).astype('Int64')

lower = movie_df['gross'].quantile(0.02)
upper = movie_df['gross'].quantile(0.98)

movie_df = movie_df[(movie_df['gross'] >= lower) & (movie_df['gross'] <= upper)]

# Xử lý missing value
movie_df["budget"].fillna(movie_df["budget"].median(), inplace=True)
movie_df["meta_score"].fillna(movie_df["meta_score"].mean(), inplace=True)
movie_df.dropna(subset=["rating", "no_of_votes", "countries", "gross"], inplace=True)

# Log-transform
movie_df['log_budget'] = np.log1p(movie_df['budget'])
movie_df['log_no_of_votes'] = np.log1p(movie_df['no_of_votes'])
movie_df['log_gross'] = np.log1p(movie_df['gross'])
movie_df['log_gross_bin'] = pd.qcut(movie_df['log_gross'], q=10, labels=False)
#

movie_df.to_csv("dataset/movies_data_processed_v4.csv", index=False)
