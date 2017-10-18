import argparse
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
from resnet_cifar10_ttq import resnet18
from squeezenet import SqueezeNet
from alexnet import AlexNet
from vgg import vgg16


parser = argparse.ArgumentParser(description='CIFAR-10')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                            help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                            help='input batch size for testing (default: 1024)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                            help='learning rate (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR',
                            help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-6, metavar='LR',
                            help='weight decay (default: 1e-6)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                            help='how many batches to wait before logging training status')
parser.add_argument('--ttq', action='store_true', default=False)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print('warning: cuda not available')

#normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
normalize = transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
train_transform = transforms.Compose((transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=2), transforms.ToTensor(), normalize))
test_transform = transforms.Compose((transforms.ToTensor(), normalize))

loader_args = {'num_workers': 4}
if use_cuda:
    loader_args.update({'pin_memory': True})

data_location = 'data/cifar10'

trainset = datasets.CIFAR10(root=data_location, train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size, **loader_args)

testset = datasets.CIFAR10(root=data_location, train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=args.test_batch_size, **loader_args)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = SqueezeNet(version=1.1, num_classes=10, small_input=True, use_ttq=args.ttq)
# model = resnet18()
# model = AlexNet(use_ttq=True, num_classes=10)

def fix_state(valid, new):
    for k, v in valid.items():
        parts = k.split('.')
        if len(parts) > 1:
            if parts[-1] == 'W_p':
                if not k in new:
                    new[k] = torch.Tensor(1)
                    new[k][0] = valid['.'.join(parts[:-1] + ['weight'])].max() / 2
            elif parts[-1] == 'W_n':
                if not k in new:
                    new[k] = torch.Tensor(1)
                    new[k][0] = valid['.'.join(parts[:-1] + ['weight'])].min() / 2
    return new

#dct = torch.load('resnet.pth')
#dct = fix_state(model.state_dict(), dct)
#model.load_state_dict(dct)

if use_cuda:
    model.cuda()

criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    running_loss = 0
    running_total = 0
    # correct = 0
    for i, (inputs, labels) in enumerate(trainloader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # forward + backward + optimize
        outputs = model(inputs)
        # pred = outputs.data.max(1, keepdim=True)[1]
        # correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        running_total += len(inputs)
        num = i + 1
        if num % args.log_interval == 0 or num == len(trainloader):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, running_total, len(trainloader.dataset),
                100 * running_total / len(trainloader.dataset), running_loss / num))

def test():
    model.eval()
    correct = 0
    running_loss = 0
    for inputs, labels in testloader:
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
        outputs = model(inputs)
        running_loss += criterion(outputs, labels, size_average=False).data[0]
        pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        running_loss / len(testloader.dataset), correct, len(testloader.dataset),
        100 * correct / len(testloader.dataset)))

for epoch in range(1, args.epochs + 1):
    start = time.time()
    train(epoch)
    delta = time.time() - start
    print('{:.2f}s/epoch'.format(delta))
    torch.save(model.state_dict(), 'model1.pth')
    test()

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = model(Variable(images.cuda(), volatile=True))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.cuda()).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
