#!/usr/bin/env python3

# skeleton from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function, division
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Function, Variable


# baseline: Average loss: 0.0989, Accuracy: 9805/10000 (98%)
# with TTQ: Average loss: 0.0710, Accuracy: 9805/10000 (98%)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                     help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: not set)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.seed:
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
data_location = 'mnist_data'
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_location, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_location, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class TTLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        self.do_reset = False    # nasty hack: superclass constructor calls reset_parameters, but defer that call until we create our parameters
        super(TTLinear, self).__init__(in_features, out_features, bias=bias)
        self.do_reset = True
        self.W_p = nn.Parameter(torch.Tensor(1))
        self.W_n = nn.Parameter(torch.Tensor(1))
        self.t = 0.05
        self.reset_parameters()

    def reset_parameters(self):
        if self.do_reset:
            super(TTLinear, self).reset_parameters()
            stdv = 1 / math.sqrt(self.weight.size(1))   # copied from superclass
            self.W_p.data.uniform_(0, stdv)
            self.W_n.data.uniform_(-stdv, 0)

    def forward(self, input):
        quantized_weight, quantized_bias = Quantize(self.t)(self.W_p, self.W_n, self.weight, self.bias)
        return F.linear(input, quantized_weight, quantized_bias)

class Quantize(Function):
    def __init__(self, t):
        super(Quantize, self).__init__()
        self.t = t

    def forward(self, W_p, W_n, *weights):
        self.save_for_backward(W_p, W_n, *weights) # save_for_backward fails if you pass in non-argument tensors
        max_weight = max(weight.abs().max() for weight in weights)
        threshold = self.t * max_weight
        p_masks = [weight.gt(threshold).float() for weight in weights]
        n_masks = [weight.lt(-threshold).float() for weight in weights]
        quantized_weights = tuple(p_masks[i] * W_p + n_masks[i] * W_n for i in range(len(weights)))
        self._threshold = threshold
        self._p_masks = p_masks
        self._n_masks = n_masks
        return quantized_weights

    def backward(self, *grad_outputs):
        # grad_outputs are gradient of loss with respect to quantized weights
        W_p, W_n, *weights = self.saved_tensors
        threshold = self._threshold
        p_masks = self._p_masks
        n_masks = self._n_masks
        z_masks = [weight.abs().le(threshold).float() for weight in weights]
        quantized_weights_ = [p_masks[i] * W_p + n_masks[i] * -W_n for i in range(len(weights))] # note the extra minus
        out = [(quantized_weights_[i] + z_masks[i]) * grad_outputs[i] for i in range(len(weights))]
        W_p_grad = W_p.clone()
        W_n_grad = W_n.clone()
        W_p_grad[0] = sum((p_masks[i] * grad_outputs[i]).sum() for i in range(len(weights)))
        W_n_grad[0] = sum((n_masks[i] * grad_outputs[i]).sum() for i in range(len(weights)))
        return (W_p_grad, W_n_grad, *out)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = TTLinear(784, 512)
        self.fc2 = TTLinear(512, 512)
        self.fc3 = TTLinear(512, 512)
        self.fc4 = TTLinear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = F.log_softmax(self.fc4(x))
        return x


model = Net()
if args.cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100 * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100 * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
