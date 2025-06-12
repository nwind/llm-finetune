import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

# scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)

num_epochs = 20

plt.plot(range(num_epochs), [scheduler(t) for t in range(num_epochs)])
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.ticklabel_format(style='plain', axis='x', useOffset=False)
plt.ylabel("Learning Rate")
plt.savefig("lr_warmup.svg")