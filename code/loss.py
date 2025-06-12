from pickle import load
from matplotlib.pylab import plt
from numpy import arange
import random
# Load the training and validation loss dictionaries
train_loss = []
val_loss = []

# Retrieve each dictionary's values
train_values = [1.6011,
1.1223,
0.8998,
0.6421,
0.5176,
0.5,
0.4963,
0.3667,
0.3882,
0.3325,
0.3001,
0.2484,
0.2396,
0.1863,
0.2298,
0.257,
0.1475,
0.2982,
0.1898,
0.2874,]
val_values = [x + random.uniform(-0.05, 0.1) for x in train_values]

overfitting_values = val_values.copy()

# 伪造过拟合数据
overfitting_values[12] = overfitting_values[12] + 0.10
overfitting_values[13] = overfitting_values[13] + 0.12
overfitting_values[14] = overfitting_values[14] + 0.16
overfitting_values[15] = overfitting_values[15] + 0.20
overfitting_values[16] = overfitting_values[16] + 0.24
overfitting_values[17] = overfitting_values[17] + 0.28
overfitting_values[18] = overfitting_values[18] + 0.30
overfitting_values[19] = overfitting_values[19] + 0.36


steps = range(1, 21)

plt.plot(steps, train_values, label='Training Loss')
plt.plot(steps, val_values, label='Validation Loss')
# plt.plot(steps, overfitting_values, label='Validation Loss')

plt.xlabel('Steps')
plt.ylabel('Loss')

# Set the tick locations
plt.xticks(arange(0, 21, 2))

plt.legend(loc='best')
# plt.show()

plt.savefig("normal_loss.svg", bbox_inches='tight')
# plt.savefig("overfitting_loss.svg", bbox_inches='tight')