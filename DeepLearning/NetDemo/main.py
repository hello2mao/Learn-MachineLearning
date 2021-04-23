# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# For data preprocess
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pprint as pp
import csv
import os


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

        feats = list(range(93))

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


def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def prep_dataloader(path, mode, batch_size, n_jobs=0):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = MyDataset(path, mode=mode)            # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)        # Construct dataloader
    return dataloader


config = {
    'tr_path': 'covid.train.csv',  # path to training data
    'tt_path': 'covid.test.csv',   # path to testing data
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 270,               # mini-batch size for dataloader
    # optimization algorithm (optimizer in torch.optim)
    'optimizer': 'SGD',
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,                 # learning rate of SGD
        'momentum': 0.9              # momentum for SGD
    },
    # early stopping epochs (the number epochs since your model's last improvement)
    'early_stop': 200,
    'save_path': 'models/model.pth'  # your model will be saved here
}

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
# get the current available device ('cpu' or 'cuda')
device = get_device()
# The trained model will be saved to ./models/
os.makedirs('models', exist_ok=True)

data_tr = pd.read_csv(config['tr_path'])  # 读取训练数据
data_tt = pd.read_csv(config['tt_path'])  # 读取测试数据
tr_set = prep_dataloader(config['tr_path'], 'train', config['batch_size'])
dv_set = prep_dataloader(config['tr_path'], 'dev', config['batch_size'])
tt_set = prep_dataloader(config['tt_path'], 'test', config['batch_size'])

# train
net = Net(tr_set.dataset.dim).to(device)  # Construct net and move to device
print(net)

loss_record = {'train': [], 'dev': []}      # for recording training loss
early_stop_cnt = 0
min_loss = 1000.
optimizer = getattr(torch.optim, config['optimizer'])(
    net.parameters(), **config['optim_hparas'])
loss_func = nn.MSELoss(reduction='mean')
for epoch in range(config['n_epochs']):
    net.train()                                 # set model to training mode
    for x, y in tr_set:                         # iterate through the dataloader
        optimizer.zero_grad()                   # set gradient to zero
        # move data to device (cpu/cuda)
        x, y = x.to(device), y.to(device)
        # forward pass (compute output)
        y_hat = net(x)
        loss = loss_func(y_hat, y)              # compute loss
        loss.backward()                         # compute gradient (backpropagation)
        optimizer.step()                        # update model with optimizer
        loss_record['train'].append(loss.detach().cpu().item())

    # After each epoch, test your model on the validation (development) set.
    net.eval()
    total_loss = 0
    for x, y in dv_set:                                        # iterate through the dataloader
        # move data to device (cpu/cuda)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():                                  # disable gradient calculation
            # forward pass (compute output)
            y_hat = net(x)
            cur_loss = loss_func(y_hat, y)                  # compute loss
        total_loss += cur_loss.detach().cpu().item() * len(x)  # accumulate loss
    # compute averaged loss
    dev_loss = total_loss / len(dv_set.dataset)
    if dev_loss < min_loss:
        # Update model if your model improved
        min_loss = dev_loss
        print('Saving model (epoch = {:4d}, loss = {:.4f})'
              .format(epoch + 1, min_loss))
        # Save model to specified path
        torch.save(net.state_dict(), config['save_path'])
        early_stop_cnt = 0
    else:
        early_stop_cnt += 1

    loss_record['dev'].append(dev_loss)
    if early_stop_cnt > config['early_stop']:
        # Stop training if your model stops improving for "config['early_stop']" epochs.
        print("Early stop")
        break

print('Finished training after {} epochs'.format(epoch))
plot_learning_curve(loss_record, title='deep model')

net.eval()
preds, targets = [], []
for x, y in dv_set:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        pred = net(x)
        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())
preds = torch.cat(preds, dim=0).numpy()
targets = torch.cat(targets, dim=0).numpy()
figure(figsize=(5, 5))
plt.scatter(targets, preds, c='r', alpha=0.5)
plt.plot([-0.2, 35], [-0.2, 35], c='b')
plt.xlim(-0.2, 35)
plt.ylim(-0.2, 35)
plt.xlabel('ground truth value')
plt.ylabel('predicted value')
plt.title('Ground Truth v.s. Prediction')
plt.show()
