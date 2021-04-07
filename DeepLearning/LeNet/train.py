from model import Model
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    batch_size = 256
    train_dataset = mnist.MNIST(root='./data', download=True, train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./data', download=True, train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = Model()
    sgd = SGD(model.parameters(), lr=1e-1)
    cross_error = CrossEntropyLoss()
    epoch = 100

    for _epoch in range(epoch):
        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = model(train_x.float())
            _error = cross_error(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, _error: {}'.format(idx, _error))
            _error.backward()
            sgd.step()

        correct = 0
        _sum = 0

        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print('accuracy: {:.2f}'.format(correct / _sum))