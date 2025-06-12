import numpy as np
import matplotlib.pyplot as plt

v = np.array([[2, 3], [2, 1], [-2, -1]])
origin = np.array([[0, 0, 0], [0, 0, 0]])

fig, ax = plt.subplots(1)
plt.quiver(*origin,
    v[:, 0], v[:, 1],
    color=['r','b','g'],
    scale=8)

ax.set_xlim((-4, 4))
ax.set_ylim((-2, 4))

# plt.show()
plt.savefig("dot_product.svg")