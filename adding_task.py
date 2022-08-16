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
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--hidden_size", type=int, default=170)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_divider", type=float, default=32)
parser.add_argument(
	"--sampler", type=str, default="LSI", choices=["LSI", "column"]
)
parser.add_argument(
	"--projector", type=str, default="projUNNT", choices=["projUNND", "projUNNT"]
)
parser.add_argument(
	"--save_name", type=str, default="adding_task_output"
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
		self.output_layer = nn.Linear(hidden_size, output_size)
		self.reset_parameters()

	def forward(self, inputs):
		hidden = None
		for input in torch.unbind(inputs, dim=1):
			hidden, output = self.rnn_layer(input, hidden)
		output = self.output_layer(hidden)
		return output

	def reset_parameters(self):
		nn.init.eye_(self.rnn_layer.recurrent_kernel.weight.data)
		# henaff_init_(self.rnn_layer.recurrent_kernel.weight.data)
		nn.init.xavier_uniform_(self.rnn_layer.input_kernel.weight.data, gain = 1.)
		nn.init.xavier_uniform_(self.output_layer.weight.data, gain = 1./2)
		# nn.init.uniform_(self.rnn_layer.nonlinearity.b.data, -0.0001,0.0001)
		# nn.init.zeros_(self.rnn_layer.nonlinearity.b.data)




# Generates Synthetic Data
def Generate_Data(size, length):
	
	# Random sequence of numbers
	x_random = np.random.uniform(0,1, size = [size, length])

	# Random sequence of zeros and ones
	x_placeholders = np.zeros((size, length))
	firsthalf = int(np.floor((length-1)/2.0))
	for i in range(0,size):
		x_placeholders[i, np.random.randint(0, firsthalf)] = 1
		x_placeholders[i, np.random.randint(firsthalf, length)] = 1

	# Create labels
	y_labels = np.reshape(np.sum(x_random*x_placeholders, axis=1), (size,1))
	
	# Creating data with dimensions (batch size, n_steps, n_input)
	data = np.dstack((x_random, x_placeholders))
	
	return torch.from_numpy(data).type(torch.float32), torch.from_numpy(y_labels).type(torch.float32)



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
		predictions = model(x_test)
		test_loss += criterion(predictions, y_test).float()
		best_loss = min(test_loss, best_loss)
	print("Test set: Average loss: {:.4f}, Best Loss: {:.4f}"
			.format(test_loss, best_loss))

	model.train()
	return test_loss,best_loss




# Network Parameters
torch_device = 'cuda:0'
input_size = 2             
hidden_size = args.hidden_size	          # Hidden layer size
output_size = 1           # One output (sum of two numbers)
n_epochs = args.epochs
batch_size = args.batch_size
train_size = 100000   # Training set size
test_size = 2000      # Testing set size
schedule_every = 1	  # scheduler applied every this amount of epochs
display_step = 100
name=args.save_name
lr = args.lr
lr_divider = args.lr_divider


# projUNN parameters
if args.sampler == "LSI":		# column_sampling_approximation or LSI_approximation
	sampler = projunn.utils.LSI_approximation 
else:
	sampler = projunn.utils.column_sampling_approximation


results = init_dict()


# n_steps = 750           # Length of sequence
for n_steps in [200,750]:

	x_train, y_train = Generate_Data(train_size, n_steps)
	x_test, y_test = Generate_Data(test_size, n_steps)

	train_data = torch.utils.data.TensorDataset(x_train,y_train)
	test_data = torch.utils.data.TensorDataset(x_test,y_test)

	#comment out below lines and revert to dataloader if test dataset is large
	x_test = x_test.to(torch_device)
	y_test = y_test.to(torch_device)

	kwargs = {'num_workers': 0, 'pin_memory': False}
	train_loader = torch.utils.data.DataLoader(
		train_data,
		batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=1000, shuffle=True, **kwargs)



	random_id = random.randint(0,99999999)
	torch.manual_seed(5544)
	np.random.seed(5544)


	model = proj_net(input_size, hidden_size, output_size).to(torch_device)
	optimizer = projunn.optimizers.RMSprop(
		model.parameters(), projector=projector, lr=args.lr
	)
	lmbda = lambda epoch: 0.94
	scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

	criterion = nn.MSELoss()

	best_loss = 999999.
	for epoch in range(n_epochs):
		processed = 0
		epoch_loss = 0

		for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
			batch_x, batch_y = batch_x.to(torch_device), batch_y.to(torch_device)

			optimizer.zero_grad()

			predictions = model(batch_x)
			loss = criterion(predictions, batch_y)


			loss.backward()

			optimizer.step()

			processed += len(batch_x)
			epoch_loss += loss.item()

			if (batch_idx % display_step) == 0:
				test_loss, best_loss = get_test_statistics(best_loss)
				results = dict_results(results)
				print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBest: {:.4f}".format(
					epoch, processed, len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.item(), best_loss))

		if scheduler is not None and (epoch % schedule_every) == 0:
			scheduler.step()

		test_loss, best_loss = get_test_statistics(best_loss)
		results = dict_results(results)
		print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBest: {:.4f}".format(
			epoch, processed, len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item(), best_loss))
		df = pd.DataFrame.from_dict(results)
		df.to_csv('data/'+name+'.csv')


