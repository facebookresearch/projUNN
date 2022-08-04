import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from torchvision import datasets, transforms
import projunn
import torchmetrics

class OrthogonalRNN(nn.Module):
    def __init__(self, hidden_size, permute):
        super(OrthogonalRNN, self).__init__()

        permute = np.random.RandomState(92916)
        self.register_buffer("permutation", torch.LongTensor(permute.permutation(784)))

        self.hidden_size = hidden_size
        self.permute = permute
        self.recurrent_kernel = nn.Linear(hidden_size, hidden_size, bias=False)
        self.input_kernel = nn.Linear(in_features=1, out_features=self.hidden_size, bias=False)
        self.lin = nn.Linear(hidden_size, 10)
        with torch.no_grad():
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")
        if args.unitary:
            # self.recurrent_kernel.weight.copy_(torch.eye(self.hidden_size))
            projunn.layers.henaff_init_(self.recurrent_kernel.weight.data)
            self.recurrent_kernel.weight.needs_projection = True

    def default_hidden(self, input):
        return input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

    def forward(self, inputs):
        if self.permute:
            inputs = inputs[:, self.permutation]
        hiddens = [self.default_hidden(inputs[:, 0, ...])]
        for input in torch.unbind(inputs, dim=1):
            out = self.input_kernel(input.unsqueeze(dim=1)) + self.recurrent_kernel(hiddens[-1])
            hiddens.append(torch.nn.functional.relu(out))
        return self.lin(hiddens[-1])

parser = argparse.ArgumentParser(description='Exponential Layer MNIST Task')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=70)
parser.add_argument('--lr', type=float, default=7e-4)
parser.add_argument("--permute", action="store_true")
parser.add_argument(
    "--optimizer", type=str, default="RMSProp", choices=["RMSProp", "SGD"]
)
parser.add_argument(
    "--projector", type=str, default="projUNND", choices=["projUNND", "projUNNT"]
)
parser.add_argument("--rank", type=int, default=1)
parser.add_argument("--unitary", action="store_true")

args = parser.parse_args()

# Fix seed across experiments
# Same seed as that used in "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform"
# https://github.com/SpartinStuff/scoRNN/blob/master/scoRNN_copying.py#L79
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(5544)
np.random.seed(5544)

# create projector and optionally the optimizer
def projector(param, update):
    a, b = projunn.utils.LSI_approximation(update/10, k=args.rank)
    if args.projector == "projUNND":
        update = projunn.utils.projUNN_D(param.data, a, b, project_on=False)
    else:
        update = projunn.utils.projUNN_T(param.data, a, b, project_on=False)
    return update

model = OrthogonalRNN(args.hidden_size, args.permute).cuda()
if args.optimizer == "RMSProp":
    optimizer = projunn.optimizers.RMSprop(
        model.parameters(), projector=projector, lr=args.lr
    )

accuracy = torchmetrics.Accuracy(compute_on_step=False).cuda()

def main():
    # Load data
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # Model and optimizers
    model.train()

    for epoch in range(100):
        accuracy.reset()
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.cuda().view(-1, 784), batch_y.cuda()
            predictions = model(batch_x)
            loss = torch.nn.functional.cross_entropy(predictions, batch_y.cuda())
            model.zero_grad()
            loss.backward()
            if args.optimizer == "RMSProp":
                optimizer.step()
            else:
                for param in model.parameters():
                    update = -args.lr * param.grad
                    if hasattr(param, "needs_projection"):
                        update = projector(update)
                    param.data.add_(update)
            accuracy.update(predictions,batch_y.cuda())
            W = model.recurrent_kernel.weight
            print("Unitary?", (W.T@W-torch.eye(W.size(1),device='cuda')).norm())
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                epoch,
                100. * batch_idx / len(train_loader), loss.item(), 100 * accuracy.compute()))

        accuracy.reset()
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.cuda().view(-1, 784)
                logits = model(batch_x)
                accuracy.update(logits, batch_y.cuda())
        print()
        print("Test set accuracy: ", accuracy.compute())

        model.train()


if __name__ == "__main__":
    main()
