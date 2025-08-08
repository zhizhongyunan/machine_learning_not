import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(20)
def generate_data(w_weight, b_weight, n):
    """
    w:[1, 2, 3]
    x[1]:[x1, x2, x3]
    """
    x = np.random.uniform(-5, 5, (n, 3))
    y = np.dot(x, w_weight) + b_weight   + np.random.normal(0, 0.5, n)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x[:, 0],x[:, 1],x[:, 2],c = y, cmap = 'viridis')
    plt.colorbar(sc)
    plt.show()
    return x, y

def loss_function(y_true, y_label):
    return np.mean((y_true - y_label) ** 2)

def partial(w, b, x, y, n):
  w_1 = 0
  w_2 = 0
  w_3 = 0
  b_partial = 0
  for i in range(n):
    error = np.dot(x[i],w) + b - y[i]
    w_1 += error * x[i][0]
    w_2 += error * x[i][1]
    w_3 += error * x[i][2]
    b_partial += error
  return np.array([w_1 / n, w_2 / n, w_3 / n]), b_partial / n

def model_train(x, y, epochs, n):
    w = np.random.uniform(-1, 1, 3)
    b = np.random.randn()
    loss_list = []
    scale = 1e-7
    a = 0.1
    pre_loss = float('inf')
    for i in range(epochs):
        y_predict = np.dot(x, w) + b
        loss = loss_function(y_predict, y)
        w_partial, b_partial = partial(w, b, x, y, n)
        w -= w_partial * a 
        b -= b_partial * a 
        if(abs(pre_loss - loss) < scale):
            break
        loss_list.append(loss)
        print(loss)
        pre_loss = loss
    return w, b, loss_list

if __name__ == '__main__':
    x, y = generate_data([0.6, 0.2, -0.4], 4, 100)
    w, b, loss_list = model_train(x, y, 1000, 100)
    print(f'final weight: {w}, bias: {b}')
    plt.plot(loss_list, color = 'blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    print(loss_list)
    plt.show()