import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
import os
import pdb

def baseline_model():
    model = Sequential()
    model.add(Dense(1000, input_dim = 23, init = 'normal', activation = 'relu'))
    model.add(Dense(1000, init = 'normal', activation = 'relu'))
    model.add(Dense(1, init = 'normal'))
    model.compile(loss = 'mean_absolute_error', optimizer = 'adam')
    return model

def load_features(file_path):
    features_df = pd.read_csv(file_path).sort_values('week_start_date')
    features_df['cty_dummy'] = features_df['city'].apply(lambda x: 1 if x == 'sj' else 0)
    # Create a scaled features matrix
    features_mat = features_df.drop(['city', 'week_start_date'], axis = 1).values
    imputer = Imputer()
    features_mat = imputer.fit_transform(features_mat)
    scaler = StandardScaler()
    features_mat = scaler.fit_transform(features_mat)
    return features_mat

def load_labels(file_path):
    labels_df = pd.read_csv(file_path)
    labels = labels_df.loc[:, 'total_cases'].values
    return labels

if __name__ == '__main__':
    train_features_path = os.path.join('data', 'dengue_features_train.csv')
    train_labels_path = os.path.join('data', 'dengue_labels_train.csv')
    test_features_path = os.path.join('data', 'dengue_features_test.csv')

    train_features = load_features(train_features_path)
    train_labels = load_labels(train_labels_path)
    test_features = load_features(test_features_path)

    mod = baseline_model()
    mod.fit(train_features, train_labels)

    predictions = mod.predict(test_features)

    # test_data = pd.read_csv(test_features_path)
    # output_df = test_data.loc[:, ['city', 'year', 'weekofyear']]
    # output_df['total_cases'] = predictions
    # output_df['total_cases'] = output_df['total_cases'].astype(int)
    # output_df.to_csv('three_layer_neural_net.csv', index = False)
