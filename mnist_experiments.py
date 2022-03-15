"""
File: mnist_experiments.py
Created Date: Wed Mar 09 2022
Author: Randall Balestriero
-----
Last Modified: Wed Mar 09 2022 10:07:14 PM
Modified By: Randall Balestriero
-----
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import torch
import torchvision
from torchvision import transforms
import torchmetrics
import argparse
import projunn
import numpy as np
import os


def load_data(name):

    # load data
    train = torchvision.datasets.__dict__[name](
        "./", download=True, train=True, transform=transforms.ToTensor()
    )
    train = torch.utils.data.random_split(train, [args.ds, len(train) - args.ds])[0]

    test = torchvision.datasets.__dict__[name](
        "./", download=True, train=False, transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=args.bs, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=args.bs, shuffle=False, drop_last=False
    )

    if name == "MNIST":
        epochs = 50
        channels = 1
        length = 28
    else:
        epochs = 90
        channels = 3
        length = 32

    return train_loader, test_loader, epochs, channels, length


parser = argparse.ArgumentParser()
parser.add_argument("-lr", type=float, default=1.0)
parser.add_argument("-lr_divider", type=float, default=10.0)
parser.add_argument("-bs", type=int, default=256)
parser.add_argument("-dname", type=str, default="CIFAR10")
parser.add_argument("-ds", type=int, default=50000)
parser.add_argument("-gamma", type=float, default=0.0)
parser.add_argument("-u", action="store_true")
args = parser.parse_args()


train_loader, test_loader, epochs, channels, length = load_data(args.dname)
model = projunn.models.ResNet9(
    channels, num_classes=10, image_length=length, unitary=args.u
).cuda()
model.train()


accuracy = torchmetrics.Accuracy(compute_on_step=False).cuda()
losses, accuracies = [], []
filename = f"logger_{args.dname}_{args.u}_{args.lr}_{args.lr_divider}_{args.bs}_{args.ds}_{args.gamma}.npz"

if os.path.exists(filename):
    import sys

    sys.exit()

if args.gamma:
    regularizer = projunn.utils.OrthoRegularizer(model, [3, 3])

for epoch in range(epochs):
    if epoch == int(epochs * 0.5) or epoch == int(epochs * 0.75):
        args.lr /= 3
    for iter, (images, labels) in enumerate(train_loader):
        prediction = model(images.cuda())
        if args.gamma:
            loss = (
                torch.nn.functional.cross_entropy(prediction, labels.cuda())
                + regularizer.regularize() * args.gamma
            )
        else:
            loss = torch.nn.functional.cross_entropy(prediction, labels.cuda())

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            accuracy(prediction, labels.cuda())
            if iter % 10 == 0:
                losses.append(loss.item())
                print(f"Current loss is: {losses[-1]}")
            for param in model.parameters():
                update = -args.lr * param.grad
                if hasattr(param, "needs_projection"):
                    a, b = projunn.utils.LSI_approximation(
                        update / args.lr_divider, k=1
                    )
                    update = projunn.utils.projUNN_T(param.data, a, b, project_on=True)
                    param.data.copy_(update)
                else:
                    param.data.add_(update)
    acc = accuracy.compute().item()
    accuracy.reset()
    accuracies.append(acc)
    print("epoch train accuracy", acc)
    model.eval()
    with torch.no_grad():
        for iter, (images, labels) in enumerate(test_loader):
            prediction = model(images.cuda())
            accuracy(prediction, labels.cuda())
    acc = accuracy.compute().item()
    accuracy.reset()
    accuracies.append(acc)
    print("epoch test accuracy", acc)
    model.train()

np.savez(
    filename,
    losses=losses,
    accuracies=accuracies,
)
