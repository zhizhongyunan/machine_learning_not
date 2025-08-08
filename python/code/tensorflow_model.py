import tensorflow as tf
import numpy as np
# x = np.array([
#     [220, 17],
#     [250, 21],
#     [120, 12],
#     [212, 18]
# ])
# y = np.array([1, 0, 0, 1])
# layer_1 = tf.keras.layers.Dense(units = 3, activation = 'sigmoid',input_shape = (2,))
# layer_2 = tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
# model = tf.keras.Sequential([layer_1, layer_2])
# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# model.fit(x, y, epochs = 1000, verbose = 1)
# predictions = model.predict(x)
# for i in range(len(predictions)):
#     print(f'Input: {x[i]}, Predicted: {predictions[i][0]:.4f}, Actual: {y[i]}')
# import tensorflow as tf
# import numpy as np

# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# def dense(a_in, W, b):
#     units = W.shape[1]
#     a_out = np.zeros(units)
#     for j in range(units):
#         w = W[:,j]
#         z = np.dot(w, a_in) + b[j]
#         a_out[j] = sigmoid(z)
#     return a_out
# W = np.array([
#     [1, -3, 4],
#     [2, 5, 7]
# ])
# b = np.array([1, 2, 3])
# x = np.array([-2, 4])
# def Sequential(x):
#     a1 = dense(x, W1, b_1)
#     a2 = dense(x, W2, b_2)
#     f_x = a_2
#     return f_x

# w = np.array([[1, 2], [3, 4]])
# print(w)

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy

x = np.array([
    [200,17],
    [250,21],
    [120,12],
    [212,18]
])
y = np.array([1, 0, 0, 1])
model = Sequential([
    Dense(units=3, activation='sigmoid', input_shape=(2,)),
    Dense(units=1, activation='sigmoid')
])
model.compile(loss=BinaryCrossentropy)
model.fit(x, y, epochs=100)
model.predict(x)