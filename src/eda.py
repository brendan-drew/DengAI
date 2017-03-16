import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_train_data():
    features_path = os.path.join('data', 'dengue_features_train.csv')
    labels_path = os.path.join('data', 'dengue_labels_train.csv')
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    return features_df, labels_df

if __name__ == '__main__':
    features_df, labels_df = load_train_data()
