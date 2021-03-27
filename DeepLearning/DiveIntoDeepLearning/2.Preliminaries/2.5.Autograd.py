import torch

x = torch.arange(4.0, requires_grad=True)
print('Default grad: ', x.grad)
y = 2 * torch.dot(x, x)
y.backward()
print('grad: ', x.grad)

x.grad.zero_()
y = x.sum()
y.backward()
print('grad: ', x.grad)

x.grad.zero_()
y = x * x
y.sum().backward()
print('grad: ', x.grad)


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print('grad: func', a.grad)
