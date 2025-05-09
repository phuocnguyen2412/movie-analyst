import os
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES =  ['log_no_of_votes', 'log_budget',
                'log_genre_stat_feature', 'release_date', 'log_country_stat_feature']
TARGET = 'log_gross'
GROSS_BIN = 8