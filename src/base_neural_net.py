import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from keras.wrappers.scikit_learn import KerasRegressor
import os
import pdb

def baseline_model():
    model = Sequential()
    model.add(Dense(1000, input_dim = 23, init = 'normal', activation = 'relu'))
    model.add(Dense(1000, init = 'normal', activation = 'relu'))
    model.add(Dense(1, init = 'normal'))
    model.compile(loss = 'mean_absolute_error', optimizer = 'adam')
    return model

def lstm_model(look_back = 1):
    model = Sequential()
    model.add(Dense(1000, input_dim = look_back))
    model.add(Dense(1))
    model.compile(loss = 'mean_absolute_error', optimizer = 'adam')
    return model

def baseline_data(features_mat, labels):
    train_size = int(features_mat.shape[0] * 0.75)
    train_index = np.arange(0,train_size)
    test_index = np.arange(train_size, features_mat.shape[0])
    train_features = features_mat[train_index,:]
    train_labels = labels[train_index]
    test_features = features_mat[test_index,:]
    test_labels = labels[test_index]
    return train_features, train_labels, test_features, test_labels

def lstm_data(labels):
    train_size = int(features_mat.shape[0] * 0.75)
    train_index = np.arange(0,train_size)
    test_index = np.arange(train_size, features_mat.shape[0])
    train_features = labels[train_index]
    train_labels = np.zeros(train_features.shape[0])
    for i, x in enumerate(train_labels):
        try:
            train_labels[i] = train_features[i + 1]
        except:
            train_labels[i] = train_features[i]
    test_features = labels[test_index]
    test_labels = np.zeros(test_features.shape[0])
    for i, x in enumerate(test_labels):
        try:
            test_labels[i] = test_features[i + 1]
        except:
            test_labels[i] = test_features[i]
    return train_features, train_labels, test_features, test_labels

if __name__ == '__main__':
    # Load features and label data
    np.random.seed(42)
    features_path = os.path.join('data', 'dengue_features_train.csv')
    labels_path = os.path.join('data', 'dengue_labels_train.csv')
    features_df = pd.read_csv(features_path).sort_values('week_start_date')
    features_df['cty_dummy'] = features_df['city'].apply(lambda x: 1 if x == 'sj' else 0)
    labels_df = pd.read_csv(labels_path)

    # Create a scaled features matrix
    features_mat = features_df.drop(['city', 'week_start_date'], axis = 1).values
    imputer = Imputer()
    features_mat = imputer.fit_transform(features_mat)
    scaler = StandardScaler()
    features_mat = scaler.fit_transform(features_mat)

    # Extract the labels
    labels = labels_df.loc[:, 'total_cases'].values

    train_features, train_labels, test_features, test_labels = baseline_data(features_mat, labels)

    # Build the neural network
    base_mod = baseline_model()
    base_mod.fit(train_features, train_labels)

    predictions = base_mod.predict(test_features)
