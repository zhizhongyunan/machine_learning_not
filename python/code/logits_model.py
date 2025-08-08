import numpy as np
import matplotlib.pyplot as plt

def averge(x):
   sum = 0
   for i in range(len(x)):
      sum += x[i]
   return sum / len(x)
def de(x, x_averge):
   sum = 0
   for i in range(len(x)):
      sum += (x[i] - x_averge) ** 2
   return np.sqrt(sum / len(x))

def generate_data(w, b, n):
  x = np.array([np.random.uniform(50, 100, n), np.random.uniform(1, 3, n), np.random.uniform(500, 2000, n)])
  x = np.array([x[0], [i*i for i in x[1]], [x[1][i]*x[2][i] for i in range(len(x[2]))]])
  x_1_E = averge(x[0])
  x_2_E = averge(x[1])
  x_3_E = averge(x[2])
  x_1_d = de(x[0], x_1_E)
  x_2_d = de(x[1], x_2_E)
  x_3_d = de(x[2], x_3_E)
  for i in range(len(x)):
    for z in range(len(x[i])):
        if i == 0:
          x[i][z] = (x[i][z] - x_1_E) / x_1_d
        elif i == 1:
          x[i][z] = (x[i][z] - x_2_E) / x_2_d
        else:
          x[i][z] = (x[i][z] - x_3_E) / x_3_d
  y = np.dot(x.T, w) + b + np.random.normal(0, 5 ,n)
  y = 1 / (1 + np.exp(-y))
  for i in range(len(y)):
    if y[i] > 0.5:
        y[i] = 1
    else:
        y[i] = 0
  plt.scatter(x[0], y, label = 'x_1_y')
  plt.legend()
  plt.xlabel('x_1')
  plt.grid(True)
  plt.show()
  return x, y
def loss_function(y_true, y_label):
    epsilon = 1e-15
    y_true = np.clip(y_true, epsilon, 1 - epsilon)
    return -np.mean(y_label * np.log(y_true) + (1 - y_label) * np.log(1 - y_true))


def partial(x, y, y_prob):
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    sum_b = 0
    y_exp = 1 / (1 + np.exp(-y_prob))
    for i in range(len(y)):
        sum_1 += (y_exp[i]-y[i]) * x[0][i]
        sum_2 += (y_exp[i]-y[i]) * x[1][i]
        sum_3 += (y_exp[i]-y[i]) * x[2][i]
        sum_b += (y_exp[i]-y[i])
    return np.array([sum_1 / len(x[0]), sum_2 / len(x[0]), sum_3 / len(x[0])]), sum_b / len(x[0])
def train_model(x, y, epochs, n):
  w = np.random.uniform(-10, 10, 3)
  b = np.random.randn()
  loss_list = []
  pre_loss = float('inf')
  step = 1e-6
  a = 0.2
  for i in range(epochs):
      y_prob = np.dot(x.T, w) + b
      print(f'epochi {i} y_prob:{y_prob}')
      loss = loss_function(y_prob, y)
      if(abs(loss - pre_loss) < step):
        break
      loss_list.append(loss)
      w_partial, b_partial = partial(x, y, y_prob)
      w -= a * w_partial
      b -= a * b_partial
      pre_loss = loss
  return w, b, loss_list
 
if __name__ == '__main__':
    x, y = generate_data(np.array([1.2, 0.4, 0.01]), 5, 100)
    w, b, loss_list = train_model(x, y, 1000, 100)
    print(w, b)
    plt.plot(loss_list, color = 'red', label = 'loss')
    plt.legend()
    plt.grid(True)
    plt.show()