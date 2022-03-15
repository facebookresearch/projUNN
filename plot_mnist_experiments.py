"""
File: plot_mnist_experiments.py
Created Date: Thu Mar 10 2022
Author: Randall Balestriero
-----
Last Modified: Thu Mar 10 2022 8:57:20 PM
Modified By: Randall Balestriero
-----
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


batch_sizes = [256]  # np.array([4, 8, 256])[[0]]
lrs = [0.5, 0.1, 0.05]  # [1.0, 0.1]
dataset_sizes = [50000]  # np.array([200, 400, 600, 10000, 50000])[[0, 1, 2]]
gammas = [
    0.0,
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    50.0,
    60.0,
    100.0,
    1000.0,
]  # , 10.0, 100.0]
dividers = [1.0, 10.0, 100.0]
dname = "CIFAR10"


cmap = matplotlib.cm.get_cmap("coolwarm")
fig, axs = plt.subplots(1, 1, sharex="all", sharey="all")
for k, gamma in enumerate(gammas):
    for i, bs in enumerate(batch_sizes):
        for j, ds in enumerate(dataset_sizes):
            for u in [True]:
                best = -np.inf
                for lr in lrs:
                    if u:
                        l = dividers
                    else:
                        l = [1.0]
                    for lr_divider in l:
                        filename = f"logger_{dname}_{u}_{lr}_{lr_divider}_{bs}_{ds}_{gamma}.npz"
                        last_train_acc = np.load(filename)["accuracies"][-1]
                        if last_train_acc > best:
                            best = last_train_acc
                            data = np.load(filename)
                axs.plot(data["accuracies"][::2] * 100, c=cmap(k / len(gammas)), ls="-")
                axs.plot(
                    data["accuracies"][1::2] * 100, c=cmap(k / len(gammas)), ls="--"
                )
if dname == "MNIST":
    plt.ylim([90, 100])
plt.savefig("test.png")
plt.close()
