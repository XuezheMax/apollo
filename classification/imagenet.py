import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import signal
import threading
from argparse import ArgumentParser
import time
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
import torch.distributed as dist
from apex.parallel import DistributedDataParallel

from optim import Apollo, RAdamW, AdaHessian
from utils import AverageMeter, accuracy


def parse_args():
    parser = ArgumentParser(description='Imagenet')
    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node. "
                             "For GPU training, this is recommended to be set to the number of GPUs in your system "
                             "so that each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed training")

    parser.add_argument('--arch', choices=['resnet18', 'resnet34', 'resnext50'], help='architecture', required=True)
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training (default: 256)')
    parser.add_argument('--eval_batch_size', type=int, default=1000, metavar='N', help='input batch size for eval (default: 1000)')
    parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: None)')
    parser.add_argument('--run', type=int, default=1, metavar='N', help='number of runs for the experiment')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--opt', choices=['sgd', 'adamw', 'radamw', 'apollo', 'adahessian'], help='optimizer', required=True)
    parser.add_argument('--lr', type=float, required=True, help='learning rate')
    parser.add_argument('--warmup_updates', type=int, default=0, metavar='N', help='number of updates to warm up (default: 0)')
    parser.add_argument('--init_lr', type=float, default=0., help='initial learning rate')
    parser.add_argument('--last_lr', type=float, default=0.01, help='last learning rate for consine lr scheduler')
    parser.add_argument('--lr_decay', choices=['exp', 'milestone', 'cosine'], required=True, help='Decay rate of learning rate')
    parser.add_argument('--milestone', type=int, nargs='+', default=[40, 80], help='Decrease learning rate at these epochs.')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--opt_h1', type=float, default=0.9, help='momentum for SGD, beta1 of Adam or beta for Apollo')
    parser.add_argument('--opt_h2', type=float, default=0.999, help='beta2 of Adam or RAdam')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight for l2 norm decay')
    parser.add_argument('--weight_decay_type', choices=['L2', 'decoupled', 'stable'], default=None, help='type of weight decay')
    parser.add_argument('--rebound', choices=['constant', 'belief'], default='constant', help='type of recified bound of diagonal hessian')
    parser.add_argument('--workers', default=6, type=int, metavar='N', help='number of data loading workers (default: 6)')
    parser.add_argument('--data_path', help='path for data file.', required=True)
    parser.add_argument('--model_path', help='path for saving model file.', required=True)
    parser.add_argument('--recover', action='store_true', help='recover the model from disk.')

    return parser.parse_args()


def is_master(rank):
    return rank <= 0


def is_distributed(rank):
    return rank >= 0


def init_distributed(model, rank, local_rank):
    print("Initializing Distributed, rank {}, local rank {}".format(rank, local_rank))
    dist.init_process_group(backend='nccl', rank=rank)
    torch.cuda.set_device(local_rank)
    return DistributedDataParallel(model)


def logging(info, logfile=None):
    print(info)
    if logfile is not None:
        print(info, file=logfile)
        logfile.flush()


def get_optimizer(opt, learning_rate, parameters, hyper1, hyper2, eps, rebound,
                  lr_decay, decay_rate, milestone, weight_decay, weight_decay_type,
                  warmup_updates, init_lr, last_lr, num_epochs, world_size):
    if opt == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=hyper1, weight_decay=weight_decay, nesterov=True)
        opt = 'momentum=%.1f, ' % (hyper1)
        weight_decay_type = 'L2'
    elif opt == 'radamw':
        optimizer = RAdamW(parameters, lr=learning_rate, betas=(hyper1, hyper2), eps=eps, weight_decay=weight_decay)
        opt = 'betas=(%.1f, %.3f), eps=%.1e, ' % (hyper1, hyper2, eps)
        weight_decay_type = 'decoupled'
    elif opt == 'adamw':
        optimizer = AdamW(parameters, lr=learning_rate, betas=(hyper1, hyper2), eps=eps, weight_decay=weight_decay)
        opt = 'betas=(%.1f, %.3f), eps=%.1e, ' % (hyper1, hyper2, eps)
        weight_decay_type = 'decoupled'
    elif opt == 'apollo':
        optimizer = Apollo(parameters, lr=learning_rate, beta=hyper1, eps=eps, rebound=rebound,
                           warmup=warmup_updates, init_lr=init_lr, weight_decay=weight_decay,
                           weight_decay_type=weight_decay_type)
        opt = 'beta=%.1f, eps=%.1e, rebound=%s, ' % (hyper1, eps, rebound)
    elif opt == 'adahessian':
        optimizer = AdaHessian(parameters, lr=learning_rate, betas=(hyper1, hyper2), eps=eps,
                               warmup=warmup_updates, init_lr=init_lr, weight_decay=weight_decay, num_threads=world_size)
        opt = 'betas=(%.1f, %.3f), eps=%.1e, ' % (hyper1, hyper2, eps)
        weight_decay_type = 'decoupled'
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

    opt += 'warmup={}, init_lr={:.1e}, wd={:.1e} ({})'.format(warmup_updates, init_lr, weight_decay, weight_decay_type)
    return optimizer, scheduler, opt


def setup(args):
    model_path = args.model_path
    args.checkpoint_name = os.path.join(model_path, 'checkpoint{}.tar'.format(args.run))
    if is_master(args.rank):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if args.recover:
            args.log = open(os.path.join(model_path, 'log.run{}.txt'.format(args.run)), 'a')
        else:
            args.log = open(os.path.join(model_path, 'log.run{}.txt'.format(args.run)), 'w')
    else:
        args.log = None

    args.cuda = torch.cuda.is_available()
    random_seed = args.seed
    if random_seed is not None:
        random_seed = random_seed + args.rank if args.rank >= 0 else random_seed
        if args.recover:
            random_seed += random.randint(0, 1024)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

    device = torch.device('cuda', args.local_rank) if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark = True
    logging("Rank {}, random seed={}: ".format(args.rank, random_seed) + str(args), args.log)

    data_path = args.data_path
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    trainset = datasets.ImageFolder(train_path,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]))
    valset = datasets.ImageFolder(val_path,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]))

    if is_master(args.rank):
        logging('Data size: training: {}, val: {}'.format(len(trainset), len(valset)))

    if args.arch == 'resnet18':
        model = models.resnet18()
    elif args.arch == 'resnet34':
        model = models.resnet34()
    elif args.arch == 'resnext50':
        model = models.resnext50_32x4d()
    else:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    model.to(device)
    args.device = device

    args.world_size = int(os.environ["WORLD_SIZE"]) if is_distributed(args.rank) else 1

    return args, (trainset, valset, len(trainset), len(valset)), model


def init_dataloader(args, trainset, valset):
    if is_distributed(args.rank):
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, rank=args.rank,
                                                                        num_replicas=args.world_size,
                                                                        shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True)

    if is_master(args.rank):
        val_loader = DataLoader(valset, batch_size=args.eval_batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
    else:
        val_loader = None
    return train_loader, train_sampler, val_loader


def single_process_main(args):
    args, (trainset, valset, num_train, num_val), model = setup(args)

    criterion = nn.CrossEntropyLoss()

    if is_master(args.rank):
        logging('# of Parameters: %d' % sum([param.numel() for param in model.parameters()]), args.log)

    if is_distributed(args.rank):
        model = init_distributed(model, args.rank, args.local_rank)

    train_loader, train_sampler, val_loader = init_dataloader(args, trainset, valset)

    epochs = args.epochs
    log = args.log

    opt = args.opt
    lr_warmup = args.warmup_updates
    init_lr = args.init_lr
    hyper1 = args.opt_h1
    hyper2 = args.opt_h2
    eps = args.eps
    rebound = args.rebound
    lr_decay = args.lr_decay
    decay_rate = args.decay_rate
    milestone = args.milestone
    weight_decay = args.weight_decay
    weight_decay_type = args.weight_decay_type
    last_lr = args.last_lr

    optimizer, scheduler, opt_param = get_optimizer(opt, args.lr, model.parameters(), hyper1, hyper2, eps, rebound,
                                                    lr_decay=lr_decay, decay_rate=decay_rate, milestone=milestone,
                                                    weight_decay=weight_decay, weight_decay_type=weight_decay_type,
                                                    warmup_updates=lr_warmup, init_lr=init_lr, last_lr=last_lr,
                                                    num_epochs=epochs, world_size=args.world_size)

    if args.recover:
        checkpoint_name = args.checkpoint_name
        print(f"Rank = {args.rank}, loading from checkpoint {checkpoint_name}")
        checkpoint = torch.load(checkpoint_name, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_top1 = checkpoint['best_top1']
        best_top5 = checkpoint['best_top5']
        best_loss = checkpoint['best_loss']
        if is_master(args.rank):
            numbers = checkpoint['numbers']
            del checkpoint
            with torch.no_grad():
                logging('Evaluating after resuming model...', log)
                eval(args, val_loader, model, criterion)
        else:
            numbers = None
            del checkpoint
    else:
        start_epoch = 1
        best_epoch = 0
        best_top1 = 0
        best_top5 = 0
        best_loss = 0
        if is_master(args.rank):
            numbers = {'train loss': [], 'train acc': [], 'test loss': [], 'test acc': []}
        else:
            numbers = None

    for epoch in range(start_epoch, epochs + 1):
        if is_distributed(args.rank):
            train_sampler.set_epoch(epoch)

        lr = scheduler.get_last_lr()[0]
        if is_master(args.rank):
            logging('Epoch: {}/{} ({}, lr={:.6f}, {})'.format(epoch, epochs, opt, lr, opt_param), log)

        train_loss, train_top1, train_top5 = train(args, train_loader, num_train, model, criterion, optimizer)
        scheduler.step()

        if is_master(args.rank):
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

            # save checkpoint
            checkpoint_name = args.checkpoint_name
            torch.save({'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_epoch': best_epoch,
                        'best_top1': best_top1,
                        'best_top5': best_top5,
                        'best_loss': best_loss,
                        'numbers': numbers},
                       checkpoint_name)


def train(args, train_loader, num_train, model, criterion, optimizer):
    model.train()
    start_time = time.time()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    num_back = 0

    device = args.device
    create_graph = args.opt == 'adahessian'

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

        loss.backward(create_graph=create_graph)

        optimizer.step()

        if step % args.log_interval == 0 and is_master(args.rank):
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            count = losses.count * args.world_size
            log_info = '[{}/{} ({:.0f}%)] loss: {:.4f}, top1: {:.2f}%, top5: {:.2f}%'.format(
                count, num_train, 100. * count / num_train,
                losses.avg, top1.avg, top5.avg)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

    all_loss = torch.Tensor([losses.avg]).to(device)
    all_top1 = torch.Tensor([top1.avg]).to(device)
    all_top5 = torch.Tensor([top5.avg]).to(device)

    if is_distributed(args.rank):
        dist.reduce(all_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(all_top1, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(all_top5, dst=0, op=dist.ReduceOp.SUM)
        all_loss = all_loss.div(args.world_size)
        all_top1 = all_top1.div(args.world_size)
        all_top5 = all_top5.div(args.world_size)

    all_loss = all_loss.item()
    all_top1 = all_top1.item()
    all_top5 = all_top5.item()
    if is_master(args.rank):
        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        logging('Average loss: {:.4f}, top1: {:.2f}%, top5: {:.2f}%, time: {:.1f}s'.format(
            all_loss, all_top1, all_top5, time.time() - start_time), args.log)

    return all_loss, all_top1, all_top5


def eval(args, val_loader, model, criterion):
    model.eval()
    if is_distributed(args.rank):
        model = model.module
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


def slurm(args):
    args_dict = vars(args)
    args_dict.pop('master_addr')
    args_dict.pop('master_port')
    args_dict.pop('nnodes')
    args_dict.pop('nproc_per_node')
    args_dict.pop('node_rank')

    current_env = os.environ
    nnodes = int(current_env['SLURM_NNODES'])
    dist_world_size = int(current_env['SLURM_NTASKS'])
    args.rank = int(current_env['SLURM_PROCID']) if dist_world_size > 1 else -1
    args.local_rank = int(current_env['SLURM_LOCALID'])

    print('start process: rank={}({}), master addr={}, port={}, nnodes={}, world size={}'.format(
        args.rank, args.local_rank, current_env["MASTER_ADDR"], current_env["MASTER_PORT"], nnodes, dist_world_size))
    current_env["WORLD_SIZE"] = str(dist_world_size)

    batch_size = args.batch_size // dist_world_size
    args.batch_size = batch_size
    single_process_main(args)


def distributed(args):
    args_dict = vars(args)

    current_env = os.environ

    nproc_per_node = args_dict.pop('nproc_per_node')
    nnodes = args_dict.pop('nnodes')
    node_rank = args_dict.pop('node_rank')

    # world size in terms of number of processes
    dist_world_size = nproc_per_node * nnodes

    # set PyTorch distributed related environmental variables
    current_env["MASTER_ADDR"] = args_dict.pop('master_addr')
    current_env["MASTER_PORT"] = str(args_dict.pop('master_port'))
    current_env["WORLD_SIZE"] = str(dist_world_size)

    batch_size = args.batch_size // dist_world_size
    args.batch_size = batch_size

    if dist_world_size == 1:
        args.rank = -1
        args.local_rank = 0
        single_process_main(args)
    else:
        mp = torch.multiprocessing.get_context('spawn')
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)

        processes = []

        for local_rank in range(0, nproc_per_node):
            # each process's rank
            dist_rank = nproc_per_node * node_rank + local_rank
            args.rank = dist_rank
            args.local_rank = local_rank
            process = mp.Process(target=run, args=(args, error_queue,), daemon=False)
            process.start()
            error_handler.add_child(process.pid)
            processes.append(process)

        for process in processes:
            process.join()


def main():
    args = parse_args()
    if 'SLURM_NNODES' in os.environ:
        slurm(args)
    else:
        distributed(args)


def run(args, error_queue):
    try:
        single_process_main(args)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.rank, traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        self.children_pids.append(pid)

    def error_listener(self):
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)


if __name__ == "__main__":
    main()
