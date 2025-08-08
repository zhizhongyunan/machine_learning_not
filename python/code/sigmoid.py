import numpy as np
import matplotlib.pyplot as plt
def generate_data(w, b, n):
  x = np.array([np.random.uniform(50, 100, n), np.random.uniform(1, 3, n), np.random.uniform(500, 2000, n)])
  x = np.array([x[0], [i*i for i in x[1]], [x[1][i]*x[2][i] for i in range(len(x[2]))]])
  y = np.dot(x.T, w) + b + np.random.normal(0, 5 ,n)
  for i in range(len(y)):
    y[i] = 1 / (1 + np.exp(-y[i]))
  for i in range(len(y)):
    if y[i] > 0.7:
        y[i] = 1
    else:
        y[i] = 0
  plt.scatter(x[0], y, label = 'x_1_y')
  plt.legend()
  plt.xlabel('x_1')
  plt.grid(True)
  plt.show()
  return x, y