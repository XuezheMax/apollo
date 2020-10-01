import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from argparse import ArgumentParser
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
from torch.optim.adamw import AdamW
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models

from optim import Apollo, ApolloW, RAdamW
from utils import AverageMeter, accuracy


def parse_args():
    parser = ArgumentParser(description='CIFAR')
    parser.add_argument('--depth', type=int, help='architecture', required=True)
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--eval_batch_size', type=int, default=1000, metavar='N', help='input batch size for eval (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: None)')
    parser.add_argument('--run', type=int, default=1, metavar='N', help='number of runs for the experiment')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--opt', choices=['sgd', 'adamw', 'radamw', 'apollo', 'apollow'], help='optimizer', required=True)
    parser.add_argument('--lr', type=float, help='learning rate', required=True)
    parser.add_argument('--warmup_updates', type=int, default=0, metavar='N', help='number of updates to warm up (default: 0)')
    parser.add_argument('--init_lr', type=float, default=0., help='initial learning rate')
    parser.add_argument('--last_lr', type=float, default=0.01, help='last learning rate for consine lr scheduler')
    parser.add_argument('--lr_decay', choices=['exp', 'milestone', 'cosine'], required=True, help='Decay rate of learning rate')
    parser.add_argument('--milestone', type=int, nargs='+', default=[80, 120], help='Decrease learning rate at these epochs.')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--opt_h1', type=float, default=0.9, help='momentum for SGD, beta1 of Adam or beta for Apollo')
    parser.add_argument('--opt_h2', type=float, default=0.999, help='beta2 of Adam or RAdam')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps of Adam')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight for l2 norm decay')
    parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')
    parser.add_argument('--data_path', help='path for data file.', required=True)
    parser.add_argument('--model_path', help='path for saving model file.', required=True)

    return parser.parse_args()


def logging(info, logfile=None):
    print(info)
    if logfile is not None:
        print(info, file=logfile)
        logfile.flush()


def get_optimizer(opt, learning_rate, parameters, hyper1, hyper2, eps, amsgrad,
                  lr_decay, decay_rate, milestone, weight_decay, warmup_updates, init_lr, last_lr, num_epochs):
    if opt == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=hyper1, weight_decay=weight_decay, nesterov=True)
        opt = 'momentum=%.1f, ' % (hyper1)
    elif opt == 'radamw':
        optimizer = RAdamW(parameters, lr=learning_rate, betas=(hyper1, hyper2), eps=eps, weight_decay=weight_decay)
        opt = 'betas=(%.1f, %.3f), eps=%.1e, ' % (hyper1, hyper2, eps)
    elif opt == 'adamw':
        optimizer = AdamW(parameters, lr=learning_rate, betas=(hyper1, hyper2), eps=eps, amsgrad=amsgrad, weight_decay=weight_decay)
        opt = 'betas=(%.1f, %.3f), eps=%.1e, amsgrad=%s, ' % (hyper1, hyper2, eps, amsgrad)
    elif opt == 'apollo':
        optimizer = Apollo(parameters, lr=learning_rate, beta=hyper1, eps=eps, warmup=warmup_updates,
                           init_lr=init_lr, weight_decay=weight_decay)
        opt = 'beta=%.1f, eps=%.1e, ' % (hyper1, eps)
    elif opt == 'apollow':
        optimizer = ApolloW(parameters, lr=learning_rate, beta=hyper1, eps=eps, warmup=warmup_updates,
                            init_lr=init_lr, weight_decay=weight_decay)
        opt = 'beta=%.1f, eps=%.1e, ' % (hyper1, eps)
    else:
        raise ValueError('unknown optimizer: {}'.format(opt))

    if lr_decay == 'exp':
        opt = opt + 'lr decay={}, decay rate={:.3f}, '.format(lr_decay, decay_rate)
        scheduler = ExponentialLR(optimizer, decay_rate)
    elif lr_decay == 'milestone':
        opt = opt + 'lr decay={} {}, decay rate={:.3f}, '.format(lr_decay, milestone, decay_rate)
        scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=decay_rate)
    elif lr_decay == 'cosine':
        opt = opt + 'lr decay={}, lr_min={}, '.format(lr_decay, last_lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=last_lr)
    else:
        raise ValueError('unknown lr decay: {}'.format(lr_decay))

    opt += 'warmup={}, init_lr={:.1e}, wd={:.1e}'.format(warmup_updates, init_lr, weight_decay)
    return optimizer, scheduler, opt


def setup(args):
    dataset = args.dataset
    data_path = args.data_path

    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    args.log = open(os.path.join(model_path, 'log.run{}.txt'.format(args.run)), 'w')

    args.cuda = torch.cuda.is_available()
    random_seed = args.seed
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark = True
    logging("Args: " + str(args), args.log)

    if dataset == 'cifar10':
        dataset = datasets.CIFAR10
        num_classes = 10
    else:
        dataset = datasets.CIFAR100
        num_classes = 100

    trainset = dataset(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]))
    valset = dataset(data_path, train=False, download=False,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      ]))

    logging('Data size: training: {}, val: {}'.format(len(trainset), len(valset)))

    model = ResNet(args.depth, num_classes=num_classes)
    model.to(device)
    args.device = device

    return args, (trainset, valset, len(trainset), len(valset)), model


def init_dataloader(args, trainset, valset):
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


def train(args, train_loader, num_train, model, criterion, optimizer):
    model.train()
    start_time = time.time()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    num_back = 0

    device = args.device

    if args.cuda:
        torch.cuda.empty_cache()

    for step, (data, y) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        outputs = model(data)
        loss = criterion(outputs, y)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, y.data, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        loss.backward()

        optimizer.step()

        if step % args.log_interval == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            log_info = '[{}/{} ({:.0f}%)] loss: {:.4f}, top1: {:.2f}%, top5: {:.2f}%'.format(
                losses.count, num_train, 100. * losses.count / num_train,
                losses.avg, top1.avg, top5.avg)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

    sys.stdout.write("\b" * num_back)
    sys.stdout.write(" " * num_back)
    sys.stdout.write("\b" * num_back)
    logging('Average loss: {:.4f}, top1: {:.2f}%, top5: {:.2f}%, time: {:.1f}s'.format(
        losses.avg, top1.avg, top5.avg, time.time() - start_time), args.log)

    return losses.avg, top1.avg, top5.avg


def eval(args, val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    device = args.device
    if args.cuda:
        torch.cuda.empty_cache()

    for data, y in val_loader:
        data = data.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        outputs = model(data)
        loss = criterion(outputs, y)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, y.data, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

    logging('Avg  loss: {:.4f}, top1: {:.2f}%, top5: {:.2f}%'.format(
        losses.avg, top1.avg, top5.avg), args.log)

    return losses.avg, top1.avg, top5.avg


def main(args):
    args, (trainset, valset, num_train, num_val), model = setup(args)

    criterion = nn.CrossEntropyLoss()

    logging('# of Parameters: %d' % sum([param.numel() for param in model.parameters()]), args.log)

    train_loader, val_loader = init_dataloader(args, trainset, valset)

    epochs = args.epochs
    log = args.log

    opt = args.opt
    lr_warmup = args.warmup_updates
    init_lr = args.init_lr
    hyper1 = args.opt_h1
    hyper2 = args.opt_h2
    eps = args.eps
    amsgrad = args.amsgrad
    lr_decay = args.lr_decay
    decay_rate = args.decay_rate
    milestone = args.milestone
    weight_decay = args.weight_decay
    last_lr = args.last_lr

    numbers = {'train loss': [], 'train acc': [], 'test loss': [], 'test acc': []}

    optimizer, scheduler, opt_param = get_optimizer(opt, args.lr, model.parameters(), hyper1, hyper2, eps, amsgrad,
                                                    lr_decay=lr_decay, decay_rate=decay_rate, milestone=milestone,
                                                    weight_decay=weight_decay, warmup_updates=lr_warmup, init_lr=init_lr,
                                                    last_lr=last_lr, num_epochs=epochs)

    best_epoch = 0
    best_top1 = 0
    best_top5 = 0
    best_loss = 0
    for epoch in range(1, epochs + 1):
        lr = scheduler.get_last_lr()[0]
        logging('Epoch: {}/{} ({}, lr={:.6f}, {})'.format(epoch, epochs, opt, lr, opt_param), log)

        train_loss, train_top1, train_top5 = train(args, train_loader, num_train, model, criterion, optimizer)
        scheduler.step()

        with torch.no_grad():
            loss, top1, top5 = eval(args, val_loader, model, criterion)

        if top1 > best_top1:
            best_top1 = top1
            best_top5 = top5
            best_loss = loss
            best_epoch = epoch

        logging('Best loss: {:.4f}, top1: {:.2f}%, top5: {:.2f}%, epoch: {}'.format(
            best_loss, best_top1, best_top5, best_epoch), args.log)

        numbers['train loss'].append(train_loss)
        numbers['test loss'].append(loss)
        numbers['train acc'].append(train_top1)
        numbers['test acc'].append(top1)
        json.dump(numbers, open(os.path.join(args.model_path, 'values.run{}.json'.format(args.run)), 'w'))


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = models.resnet.BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = models.resnet.Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    args = parse_args()
    main(args)
