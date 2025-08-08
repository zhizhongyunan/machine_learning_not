import numpy as np
import matplotlib.pyplot as plt
import os 
"""
    author:yanzhengqi
    date:2025/7/18
    description:一元线性回归模型
    1. 生成数据
    2. 定义损失函数 
    3. 梯度下降
    4. 绘图
"""
np.random.seed(10)
def generate_date(w_weight, b_weight, n):
    x = np.random.uniform(-10, 10, n)
    y = w_weight * x + b_weight + np.random.normal(0, 5, n)
    plt.scatter(x, y, label = 'data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()
    with open(os.path.join(os.getcwd(),'python/data.txt'),'w') as f:
        for i in range(n):
            f.write(f'{x[i]},{y[i]}\n')
    return x, y
def loss_function(y_true, y_label, n):
  """sum_loss = 0
  for i in range(n):
    sum_loss += np.pow((y_true[i] - y_label[i]), 2)
  return sum_loss / n"""
  return np.mean((y_true - y_label) ** 2)

def partial(w, b, x, y):
  w_sum = 0
  b_sum = 0
  for i in range(len(x)):
    w_sum += (w * x[i] + b - y[i]) * x[i]
    b_sum += (w * x[i] + b - y[i])
  return w_sum / len(x), b_sum / len(x)

def linear_model(epochs, x, y, n):
    
    w = np.random.randn()
    b = np.random.randn()
    print(f'begin weight:{w}, bias: {b}')
    a = 0.01
    loss_list = []
    pre_loss = float("inf")
    stop_scale = 1e-5
    for i in range(epochs):
        y_true = w * x + b
        loss = loss_function(y_true, y, n)
        loss_list.append(loss)
        if(abs(pre_loss - loss) < stop_scale):
            break
        pre_loss = loss
        w_partial, b_partial= partial(w, b, x, y)
        w -= a * w_partial
        b -= a * b_partial

    return w, b, loss_list

def plot_loss(loss_list):
    plt.plot(loss_list, label = 'loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

def plot_model(x, y, w, b):
    plt.scatter(x, y, label = 'data')
    plt.plot(x, w * x + b, color = 'red', label = 'model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    x, y = generate_date(2, 3, 1000)
    w, b, loss_list = linear_model(10, x, y, 1000)
    plot_loss(loss_list)
    plot_model(x, y, w, b)
    print(f'weight:{w}, bias:{b}')
    