import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

def baseline_data(features_mat, labels):
    train_size = int(features_mat.shape[0] * 0.75)
    train_index = np.arange(0,train_size)
    test_index = np.arange(train_size, features_mat.shape[0])
    train_features = features_mat[train_index,:]
    train_labels = labels[train_index]
    test_features = features_mat[test_index,:]
    test_labels = labels[test_index]
    return train_features, train_labels, test_features, test_labels

if __name__ == '__main__':
    # Load features and label data
    np.random.seed(42)
    features_path = os.path.join('data', 'dengue_features_train.csv')
    labels_path = os.path.join('data', 'dengue_labels_train.csv')
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    # Subset data from San Jose
    sj_features = features_df.loc[features_df['city'] == 'sj', :]
    sj_labels = labels_df.loc[labels_df['city'] == 'sj']

    # Create a scaled features matrix
    features_mat = sj_features.drop(['city', 'week_start_date', 'year'], axis = 1).values
    imputer = Imputer()
    features_mat = imputer.fit_transform(features_mat)
    scaler = StandardScaler()
    features_mat = scaler.fit_transform(features_mat)

    # Extract the labels
    labels = labels_df.loc[labels_df['city'] == 'sj', 'total_cases'].values

    train_features, train_labels, test_features, test_labels = baseline_data(features_mat, labels)

    gb = GradientBoostingRegressor(max_features = 5)
    gb.fit(train_features, train_labels)
    predictions = gb.predict(test_features)
    plt.plot(range(len(predictions)),test_labels)
    plt.plot(range(len(predictions)),predictions, color = 'r')
    plt.show()
