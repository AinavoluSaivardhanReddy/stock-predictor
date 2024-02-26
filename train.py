from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from processing import create_dataset, process_data
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


data = yf.download('AAPL', start='2010-01-01', end='2021-01-01')
# data = yf.downl oad('TSLA', start='2015-01-01', end='2021-01-01')

train_data, test_data, scaled_data, scaler = process_data(data, 0.7)
time_step = 90
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create the LSTM model
model = Sequential()
model.add(LSTM(80, activation='relu', return_sequences=True, input_shape=(time_step, 1)))
# model.add(LSTM(80, activation='relu', return_sequences=True))
model.add(LSTM(40, return_sequences=False))
model.add(Dense(20))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=2, epochs=5)

