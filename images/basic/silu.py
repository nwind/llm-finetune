import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

matplotlib.rcParams['font.family'] = ['LiHei Pro', 'sans-serif']


def swish(x, beta=1):
   return x * (1 / (1 + np.exp(-beta * x)))

x_values = np.linspace(-6, 6, 500)
y_values = swish(x_values)

swish_values = swish(x_values)

plt.plot(x_values, swish_values)
plt.title("SiLU 激活函数")
plt.xlabel("输入值")
plt.ylabel("输出值")
plt.grid()
plt.legend()
plt.show()
