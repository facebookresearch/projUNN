import torch
import torch.nn as nn
import torch.nn.functional as F
import projunn
import argparse
import sys

import numpy as np
import time
from torchvision import datasets, transforms
import time
import random
import pandas as pd


parser = argparse.ArgumentParser(description="Adding Task")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=190)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=7e-4)
parser.add_argument("--lr_divider", type=float, default=32)
parser.add_argument(
	"--sampler", type=str, default="LSI", choices=["LSI", "column"]
)
parser.add_argument(
	"--projector", type=str, default="projUNNT", choices=["projUNND", "projUNNT"]
)
parser.add_argument(
	"--save_name", type=str, default="copy_task_output"
)
parser.add_argument("--rank", type=int, default=1)

args = parser.parse_args()





# create projector and optionally the optimizer
def projector(param, update):
	a, b = sampler(update/lr_divider, k=args.rank)
	if args.projector == "projUNND":
		update = projunn.utils.projUNN_D(param.data, a, b, project_on=False)
	else:
		update = projunn.utils.projUNN_T(param.data, a, b, project_on=False)
	return update




class proj_net(nn.Module):

	def __init__(self, input_size, hidden_size, output_size):
		super(proj_net, self).__init__()
		self.rnn_layer = projunn.layers.OrthogonalRNN(input_size, hidden_size,nonlinearity = nn.ReLU)
		self.output_layer = nn.Linear(hidden_size, output_size, bias = True)
		self.reset_parameters()

	def forward(self, inputs):
		hidden = None
		outputs = []
		for input in torch.unbind(inputs, dim=1):
			hidden, output = self.rnn_layer(input, hidden)
			outputs.append(self.output_layer(hidden))
		return torch.stack(outputs, dim=1)

	def reset_parameters(self):
		projunn.layers.henaff_init_(self.rnn_layer.recurrent_kernel.weight.data)
		# nn.init.eye_(self.rnn_layer.recurrent_kernel.weight.data)
		nn.init.xavier_uniform_(self.rnn_layer.input_kernel.weight.data, gain = 1.)
		nn.init.kaiming_normal_(self.output_layer.weight.data, nonlinearity="relu")
		nn.init.constant_(self.output_layer.bias.data, 0)
		# nn.init.uniform_(self.rnn_layer.nonlinearity.b.data, -0.0001,0.0001)
		# nn.init.zeros_(self.rnn_layer.nonlinearity.b.data)



# Generates Synthetic Data
def Generate_Data(L, K, batch_size):
    seq = np.random.randint(1, high=9, size=(batch_size, K))
    zeros1 = np.zeros((batch_size, L))
    zeros2 = np.zeros((batch_size, K-1))
    zeros3 = np.zeros((batch_size, K+L))
    marker = 9 * np.ones((batch_size, 1))

    x = torch.LongTensor(np.concatenate((seq, zeros1, marker, zeros2), axis=1))
    y = torch.LongTensor(np.concatenate((zeros3, seq), axis=1))

    return x, y

def init_dict():
	keys = ['random_id',
			'lr_default',
			'lr_polar',
			'rank_k',
			'batch_size',
			'epoch',
			'step',
			'best_loss',
			'test_loss',
			'train_loss',
			'n_epochs',
			'hidden_size',
			'sequence_length',
			'name']
	return {new_list: [] for new_list in keys}

def dict_results(results):
	results['random_id'].append(random_id)
	results['lr_default'].append(lr)
	results['lr_polar'].append(lr/lr_divider)
	results['rank_k'].append(args.rank)
	results['batch_size'].append(batch_size)
	results['epoch'].append(epoch)
	results['step'].append(batch_idx)
	results['best_loss'].append(best_loss.item())
	results['test_loss'].append(test_loss.item())
	results['train_loss'].append(epoch_loss/(batch_idx+1))
	results['n_epochs'].append(n_epochs)
	results['hidden_size'].append(hidden_size)
	results['sequence_length'].append(n_steps)
	results['name'] = name
	return results



def get_test_statistics(best_loss):
	model.eval()
	with torch.no_grad():
		test_loss = 0.
		# for test_batch_x, test_batch_y in test_loader:
		# 	test_batch_x, test_batch_y = test_batch_x.to(torch_device), test_batch_y.to(torch_device)
		# 	predictions = model(test_batch_x)
		# 	test_loss += criterion(predictions, test_batch_y).float()
		# test_loss /= len(test_loader)

		predictions = model(x_test_onehot)
		test_loss += criterion(predictions.view(-1,n_classes), y_test.view(-1)).float()
		best_loss = min(test_loss, best_loss)
	print("Test set: Average loss: {:.4f}, Best Loss: {:.4f}"
			.format(test_loss, best_loss))

	model.train()
	return test_loss,best_loss

def onehot(input):
    return F.one_hot(input,input_size).type(torch.float32)



# Network Parameters
torch_device = 'cuda:0'
n_classes = 9
K = 10
input_size = n_classes + 1             
hidden_size = args.hidden_size          # Hidden layer size
output_size = n_classes    
n_epochs = args.epochs
batch_size = args.batch_size
schedule_every = 1	  # scheduler applied every this amount of epochs
display_step = 100
name=args.save_name
iterations = 4000
train_size = iterations *  batch_size   # Training set size
test_size = 100      # Testing set size
lr = args.lr
lr_divider = args.lr_divider




# projUNN parameters
if args.sampler == "LSI":		# column_sampling_approximation or LSI_approximation
	sampler = projunn.utils.LSI_approximation 
else:
	sampler = projunn.utils.column_sampling_approximation



results = init_dict()


for n_steps in [1000,2000]:
	n_len = n_steps + 2 * K


	x_train, y_train = Generate_Data(n_steps,K,train_size)
	x_test, y_test = Generate_Data(n_steps,K,test_size)

	#comment out below lines and revert to dataloader if test dataset is large
	x_test = x_test.to(torch_device)
	x_test_onehot = onehot(x_test)
	y_test = y_test.to(torch_device)

	# train_data = torch.utils.data.TensorDataset(x_train,y_train)
	# test_data = torch.utils.data.TensorDataset(x_test,y_test)

	# kwargs = {'num_workers': 0, 'pin_memory': False}
	# train_loader = torch.utils.data.DataLoader(
	# 	train_data,
	# 	batch_size=batch_size, shuffle=True, **kwargs)
	# test_loader = torch.utils.data.DataLoader(
	# 	test_data,
	# 	batch_size=1000, shuffle=True, **kwargs)




	random_id = random.randint(0,99999999)
	torch.manual_seed(5544)
	np.random.seed(5544)


	model = proj_net(input_size, hidden_size, output_size).to(torch_device)
	optimizer = projunn.optimizers.RMSprop(
		model.parameters(), projector=projector, lr=args.lr
	)
	lmbda = lambda epoch: 0.9
	scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

	criterion = nn.CrossEntropyLoss()

	best_loss = 999999.
	for epoch in range(n_epochs):
		processed = 0
		epoch_loss = 0

		for batch_idx in range(iterations):
			batch_x = x_train[batch_idx * batch_size : (batch_idx+1) * batch_size].to(torch_device)
			batch_y = y_train[batch_idx * batch_size : (batch_idx+1) * batch_size].to(torch_device)
			batch_x_onehot = onehot(batch_x)

			optimizer.zero_grad()

			predictions = model(batch_x_onehot)
			loss = criterion(predictions.view(-1,n_classes), batch_y.view(-1))

			loss.backward()

			optimizer.step()

			processed += len(batch_x)
			epoch_loss += loss.item()

			if (batch_idx % display_step) == 0:
				test_loss, best_loss = get_test_statistics(best_loss)
				results = dict_results(results)
				print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBest: {:.4f}".format(
									epoch, processed, train_size,
									100. * batch_idx / iterations, loss.item(), best_loss))

		if scheduler is not None and (epoch % schedule_every) == 0:
			scheduler.step()

		test_loss, best_loss = get_test_statistics(best_loss)
		results = dict_results(results)
		df = pd.DataFrame.from_dict(results)
		df.to_csv('data/'+name+'.csv')


