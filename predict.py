from train import model, X_train, X_test, time_step, scaler, scaled_data, y_test
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_test, test_predict)
print("error:", math.sqrt(mse))

trainPredictPlot = np.empty_like(scaled_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict) + time_step, :] = train_predict

testPredictPlot = np.empty_like(scaled_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

plt.plot(scaler.inverse_transform(scaled_data))
plt.plot(trainPredictPlot, label="trained data", color="blue")
plt.plot(testPredictPlot, label="test data", color="green")
plt.show()
