import pandas as pd


import numpy as np
def convert_votes(vote):
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
import re
def convert_released_day(date):
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

movie_df = pd.read_csv('movies_data.csv')

movie_df = movie_df.drop_duplicates(subset=["name"], keep="first")

movie_df["gross"] = movie_df["gross"].apply(convert_gross_budget).astype('float')

movie_df["budget"]= movie_df["budget"].apply(convert_gross_budget).astype('float')

movie_df["ROI"] = movie_df.apply(lambda x: (x["gross"] - x["budget"]) / x["budget"] if x["budget"] != 0 else np.nan, axis=1)

movie_df["release_date"] = movie_df["release_date"].apply(convert_released_day).astype("str")

movie_df["no_of_votes"] = movie_df["no_of_votes"].apply(convert_votes).astype('Int64')

movie_df["result"] = movie_df.apply(make_success_label, axis=1)

movie_df.to_csv("movies_data_processed.csv", index=False)