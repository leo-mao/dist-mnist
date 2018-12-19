import argparse
import torch.nn as nn
import time

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--backend', type=str, default='gloo')
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world-size', type=int, default=4)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('----Torch Config----')
print('mini batch-size : {}'.format(args.batch_size))
print('world-size : {}'.format(args.world_size))
print('backend : {}'.format(args.backend))
print('--------------------')
# world_size is the number of processes
dist.init_process_group(backend=args.backend, world_size=args.world_size, group_name='pytorch_test',
                        rank=args.rank)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    # torch.cuda.set_device(args.local_rank)
else:
    device = torch.device('cpu')

train_dataset = datasets.MNIST('../MNIST_data/', train=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize((0.1307,), (0.3081,))]))

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size,
                                                                rank=args.rank)

kwargs = {'num_workers': args.world_size, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../MNIST_data/', train=False,
                                                         transform=transforms.Compose(
                                                             [transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                          batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)


model = Net()
cudnn.benchmark = True

if args.cuda:
    model.cuda(device=device)
    model = nn.parallel.DataParallel(model)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


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
            print('Train Epoch {} - {} / {:3.0f} \tLoss  {:.6f}'.format(
                epoch, batch_idx, 1.0 * len(train_loader.dataset) / len(data), loss))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # Varibale(data, volatile=True)
        data, target = Variable(data, requires_grad=False), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum')
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))


tot_time = 0
start_wall_time = time.time()
for epoch in range(1, args.epochs + 1):
    train_sampler.set_epoch(epoch)
    start_cpu_secs = time.time()
    train(epoch)
    end_cpu_secs = time.time()
    # print('start_cpu_secs {}'.format())
    print("Epoch {} of took {:.3f}s".format(
        epoch, end_cpu_secs - start_cpu_secs))

    tot_time += end_cpu_secs - start_cpu_secs
    print('Current Total time : {:.3f}s'.format(tot_time))

test()

print("Total time= {:.3f}s".format(tot_time))