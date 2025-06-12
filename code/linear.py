import numpy as np
import matplotlib.pyplot as plt


lr_min = 0.001
lr_max = 0.1
max_epochs = 100

# Generate learning rate schedule
lr = [
    lr_min + 0.5 * (lr_max - lr_min) * (1 - epoch / max_epochs)
    for epoch in range(max_epochs)
]

# Plot
plt.figure(figsize=(8, 8))
plt.plot(lr)
plt.savefig("linear.svg")