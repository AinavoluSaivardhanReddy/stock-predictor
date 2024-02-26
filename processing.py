import numpy as np
from sklearn.preprocessing import MinMaxScaler


def process_data(data, training_ratio):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    training_size = int(len(scaled_data) * training_ratio)
    test_size = len(scaled_data) - training_size
    train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :1]
    return train_data, test_data, scaled_data, scaler


# Create a dataset for LSTM
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

