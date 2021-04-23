# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For plotting
import matplotlib.pyplot as plt

# For data preprocess
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    ''' Dataset for loading and preprocessing the dataset '''

    def __init__(self,
                 path,
                 mode='train'):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        # feats = list(range(93))
        feats = list(range(40)) + [57, 75]

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]

            # Splitting training data into train & dev sets
            indices_tr, indices_dev = train_test_split([i for i in range(data.shape[0])], test_size = 0.3, random_state = 0)
            if mode == 'train':
                indices = indices_tr
            elif mode == 'dev':
                indices = indices_dev

            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


class Net(nn.Module):
    ''' A simple deep neural network '''

    def __init__(self, input_dim):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def prep_dataloader(path, mode, batch_size, n_jobs=0):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = MyDataset(path, mode=mode)            # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)        # Construct dataloader
    return dataloader


myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
# get the current available device ('cpu' or 'cuda')
device = get_device()

tr_path = 'covid.train.csv'
batch_size = 270
data_tr = pd.read_csv(tr_path)
tr_set = prep_dataloader(tr_path, 'train', batch_size)

# nets
net_SGD = Net(tr_set.dataset.dim).to(device)
net_Momentum = Net(tr_set.dataset.dim).to(device)
net_RMSprop = Net(tr_set.dataset.dim).to(device)
net_Adam = Net(tr_set.dataset.dim).to(device)
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# optimizers
learning_rate = 0.001
opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=learning_rate)
opt_Momentum = torch.optim.SGD(
    net_Momentum.parameters(), lr=learning_rate, momentum=0.8, nesterov=True)
opt_RMSprop = torch.optim.RMSprop(
    net_RMSprop.parameters(), lr=learning_rate, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(),
                            lr=learning_rate, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

n_epochs = 3000
loss_func = nn.MSELoss(reduction='mean')
loss_histories = [[], [], [], []]
for epoch in range(n_epochs):
    # iterate through the dataloader
    for step, (x, y) in enumerate(tr_set):
        for net, optimizer, loss_history in zip(nets, optimizers, loss_histories):
            optimizer.zero_grad()                   # set gradient to zero
            # move data to device (cpu/cuda)
            x, y = x.to(device), y.to(device)
            # forward pass (compute output)
            y_hat = net(x)
            loss = loss_func(y_hat, y)              # compute loss
            loss.backward()                         # compute gradient (backpropagation)
            optimizer.step()                        # update model with optimizer
            loss_history.append(loss.data.numpy())
        if step % 50 == 0 and epoch % 50 == 0:
            print(
                'epoch: {:4d}, SGD: {:.4f}, Momentum: {:.4f}, RMSprop: {:.4f}, Adam: {:.4f}'.format(epoch, loss_histories[0][-1],
                                                                                                    loss_histories[1][-1],
                                                                                                    loss_histories[2][-1],
                                                                                                    loss_histories[3][-1]))

print('Finished training after {} epochs'.format(epoch))
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam', ]
for i, loss_history in enumerate(loss_histories):
    plt.plot(loss_history, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 15.0))
plt.xlim((0, len(tr_set.dataset)))
print('epoch: {}/{},steps:{}/{}'.format(epoch+1,
      n_epochs, step*batch_size, len(tr_set.dataset)))
plt.show()
