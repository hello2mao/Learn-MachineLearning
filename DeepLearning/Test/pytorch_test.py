import torch
from torch.autograd import Variable

if __name__ == '__main__':
    x = Variable(torch.ones(2, 2), requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    print(z, out)
    out.backward()
    print(x.grad)