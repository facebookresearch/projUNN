import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from torchvision import datasets, transforms
import projunn
import torchmetrics

parser = argparse.ArgumentParser(description="Exponential Layer MNIST Task")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=170)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--permute", action="store_true")
parser.add_argument(
    "--optimizer", type=str, default="RMSProp", choices=["RMSProp", "SGD"]
)
parser.add_argument(
    "--projector", type=str, default="projUNND", choices=["projUNND", "projUNNT"]
)
parser.add_argument("--rank", type=int, default=1)

args = parser.parse_args()

# Fix seed across experiments
# Same seed as that used in "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform"
# https://github.com/SpartinStuff/scoRNN/blob/master/scoRNN_copying.py#L79
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(5544)
np.random.seed(5544)
permute = np.random.RandomState(92916)
permutation = torch.LongTensor(permute.permutation(784))

# create projector and optionally the optimizer
def projector(param, update):
    a, b = projunn.utils.LSI_approximation(update, k=args.rank)
    if args.projector == "projUNND":
        update = projunn.utils.projUNN_D(param.data, a, b, project_on=False)
    else:
        update = projunn.utils.projUNN_T(param.data, a, b, project_on=False)
    return update


class proj_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(proj_net, self).__init__()
        self.rnn_layer = projunn.layers.OrthogonalRNN(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        hidden = None
        for input in torch.unbind(inputs, dim=1):
            hidden, output = self.rnn_layer(input, hidden)
        output = self.output_layer(hidden)
        return output




def main():
    # Load data
    kwargs = {"num_workers": 1, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./mnist", train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./mnist", train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )



    # Model and optimizers
    model = proj_net(1, args.hidden_size, 10).cuda()
    model.train()
    if args.optimizer == "RMSProp":
        optimizer = projunn.optimizers.RMSprop(
            model.parameters(), projector=projector, lr=args.lr
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 60*len(train_loader), 0.2)

    accuracy = torchmetrics.Accuracy(compute_on_step=False).cuda()


    for epoch in range(100):
        accuracy.reset()
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if args.permute:
                batch_x = batch_x.cuda().view(-1, 784, 1)[:, permutation]
            else:
                batch_x = batch_x.cuda().view(-1, 784, 1)
            predictions = model(batch_x)
            loss = torch.nn.functional.cross_entropy(predictions, batch_y.cuda())
            model.zero_grad()
            loss.backward()
            if args.optimizer == "RMSProp":
                optimizer.step()
                scheduler.step()
            else:
                for param in model.parameters():
                    update = -args.lr * param.grad
                    if hasattr(param, "needs_projection"):
                        update = projector(param, update)
                    param.data.add_(update)
            accuracy.update(predictions, batch_y.cuda())
            W = model.rnn_layer.recurrent_kernel.weight
            print("Unitary?", (W.T @ W - torch.eye(W.size(1), device="cuda")).norm())
            print(
                "Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}\tAccuracy: {:.2f}%".format(
                    epoch,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    100 * accuracy.compute(),
                )
            )

        accuracy.reset()
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                if args.permute:
                    batch_x = batch_x.cuda().view(-1, 784, 1)[:, permutation]
                else:
                    batch_x = batch_x.cuda().view(-1, 784, 1)
                logits = model(batch_x)
                accuracy.update(logits, batch_y.cuda())
        print()
        print("Test set accuracy: ", accuracy.compute())

        model.train()


if __name__ == "__main__":
    main()
