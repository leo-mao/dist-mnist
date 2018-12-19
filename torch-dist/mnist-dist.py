import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.optim as optim

from random import Random
from torchvision import datasets, transforms
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
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
parser.add_argument('--backend', type=str, default='nccl')
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world-size', type=int, default=1)
parser.add_argument('--local_rank', type=int)
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


class Partition(object):
    """ Dataset-like object, but only access a subset of it"""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rand = Random()
        rand.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rand.shuffle(indexes)

        for fraction in sizes:
            part_len = int(fraction * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


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


def partition_dataset():
    """ Partitioning MNIST"""
    dataset = datasets.MNIST(
        '../MNIST_data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    size = dist.get_world_size()
    batch_size = 1.0 * 128 / size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=batch_size, shuffle=True)
    return train_set, batch_size


def average_gradients(model):
    """ Gradient averaging"""
    # group = dist.new_group([x for x in range(args.world_size)])
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def cal_print_summary(rank, loss, accuracy):
    group = dist.new_group([x for x in range(args.world_size)])
    size = float(dist.get_world_size())
    summaries = torch.tensor([loss, accuracy], requires_grad=False, device='cuda')
    dist.reduce(summaries, 0, op=dist.ReduceOp.SUM)
    if rank == 0:
        summaries /= size
        print('\nSystem : Average loss: {:.4f}, Average Accuracy: {:.2f}%\n'.format(summaries[0], summaries[1] * 100))


def train(model, optimizer, train_loader, epoch):
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

        average_gradients(model)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch {} - {} / {:3.0f} \tLoss  {:.6f}'.format(
                epoch, batch_idx, 1.0 * len(train_loader.dataset) / len(data), loss))


def test(test_loader, model):
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
    print('\nTest set : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))
    return test_loss, float(correct) / len(test_loader.dataset)


def run(rank, batch_size, world_size):
    """ Distributed Synchronous SGD Example """

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda')
        # torch.cuda.set_device(args.local_rank)
    else:
        device = torch.device('cpu')

    train_dataset = datasets.MNIST('../MNIST_data/', train=True,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size,
                                                                    rank=rank)

    kwargs = {'num_workers': args.world_size, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../MNIST_data/', train=False,
                                                             transform=transforms.Compose(
                                                                 [transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,), (0.3081,))])),
                                              batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net()
    cudnn.benchmark = True

    if args.cuda:
        model.cuda(device=device)
        model = nn.parallel.DataParallel(model)
        # model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    torch.manual_seed(args.seed)
    tot_time = 0
    first_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        start_cpu_secs = time.time()
        train(model, optimizer, train_loader, epoch)
        end_cpu_secs = time.time()
        # print('start_cpu_secs {}'.format())
        print("Epoch {} of took {:.3f}s".format(
            epoch, end_cpu_secs - start_cpu_secs))

        tot_time += end_cpu_secs - start_cpu_secs
        print('Current Total time : {:.3f}s'.format(tot_time))
        if epoch == 1:
            first_epoch = tot_time

    test_loss, accuracy = test(test_loader, model)

    if args.epochs > 1:
        print('Average epoch time(ex. 1.) : {:.3f}s'.format(float(tot_time - first_epoch)
                                                            / (args.epochs - 1)))
    print("Total time : {:.3f}s".format(tot_time))
    cal_print_summary(rank, test_loss, accuracy)


if __name__ == '__main__':
    run(args.rank, args.batch_size, args.world_size)
