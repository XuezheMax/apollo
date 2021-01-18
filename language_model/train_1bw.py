import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import argparse
import random
import pickle
import math
import json
import numpy as np
import torch
from utils import clip_grad_norm_
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from language_model.model_word_ada.lm import LM
from language_model.model_word_ada.dataset import LargeDataset, EvalDataset
from optim import RAdamW, Apollo, AdaHessian, AdaBelief


def logging(info, logfile=None):
    print(info)
    if logfile is not None:
        print(info, file=logfile)
        logfile.flush()


def get_optimizer(opt, learning_rate, parameters, lr_decay, decay_rate, milestone, warmup_updates, init_lr, rebound):
    if opt == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=0., nesterov=True)
    elif opt == 'radam':
        optimizer = RAdamW(parameters, lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.)
    elif opt == 'adam':
        optimizer = Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.)
    elif opt == 'adabelief':
        optimizer = AdaBelief(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-12, weight_decay=0.)
    elif opt == 'apollo':
        optimizer = Apollo(parameters, lr=learning_rate, beta=0.9, eps=1e-4, rebound=rebound,
                           warmup=warmup_updates, init_lr=init_lr, weight_decay=0.)
    elif opt == 'adahessian':
        optimizer = AdaHessian(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-4,
                               warmup=warmup_updates, init_lr=init_lr, weight_decay=0.)
    else:
        raise ValueError('unknown optimizer: {}'.format(opt))

    opt_param = 'lr decay={} {}, decay rate={:.3f}'.format(lr_decay, milestone, decay_rate)
    scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=decay_rate)

    if opt == 'apollo':
        opt_param += ', rebound={}'.format(rebound)
    if opt in ['apollo', 'adahessian']:
        opt_param += ', warmup={}, init_lr={:.1e}'.format(warmup_updates, init_lr)
    return optimizer, scheduler, opt_param


def evaluate(args, data_loader, lm_model):
    logging('evaluating', args.log)
    lm_model.eval()

    iterator = data_loader.get_tqdm()
    hx = None
    device = args.device
    total_loss = 0
    total_count = 0
    for word_t, label_t in iterator:
        word_t = word_t.to(device)
        label_t = label_t.to(device).view(-1)
        count = label_t.size(0)
        loss, hx = lm_model(word_t, label_t, hx=hx)
        total_loss += count * loss.item()
        total_count += count

    ppl = math.exp(total_loss / total_count)
    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', default='data/billionwords/one_billion/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=20)
    parser.add_argument('--hid_dim', type=int, default=2048)
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--clip', type=float, default=0)
    parser.add_argument('--clip_mode', choices=['total', 'each'], default='total')
    parser.add_argument('--opt', choices=['sgd', 'adam', 'radam', 'adabelief', 'apollo', 'adahessian'], help='optimizer', required=True)
    parser.add_argument('--rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--lr_decay', choices=['milestone'], default='milestone', help='Decay rate of learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--milestone', type=int, nargs='+', default=[12, 18], help='Decrease learning rate at these epochs.')
    parser.add_argument('--rebound', choices=['constant', 'belief'], default='constant', help='type of recified bound of diagonal hessian')
    parser.add_argument('--warmup_updates', type=int, default=0, metavar='N', help='number of updates to warm up (default: 0)')
    parser.add_argument('--init_lr', type=float, default=0, help='initial learning rate')
    parser.add_argument('--cutoffs', nargs='+', default=[60000, 100000, 640000])
    parser.add_argument('--interval', type=int, default=1000)
    parser.add_argument('--model_path', help='path for saving model file.', required=True)
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: None)')
    parser.add_argument('--run', type=int, default=1, metavar='N', help='number of runs for the experiment')
    parser.add_argument('--recover', action='store_true', help='recover the model from disk.')
    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if args.recover:
        args.log = open(os.path.join(model_path, 'log.run{}.txt'.format(args.run)), 'a')
    else:
        args.log = open(os.path.join(model_path, 'log.run{}.txt'.format(args.run)), 'w')
    args.checkpoint_name = os.path.join(model_path, 'checkpoint{}.tar'.format(args.run))

    args.cuda = torch.cuda.is_available()
    random_seed = args.seed
    if random_seed is not None:
        if args.recover:
            random_seed += random.randint(0, 1024)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    if args.cuda:
        torch.cuda.set_device(device)

    if args.opt == 'adahessian':
        torch.backends.cudnn.enabled = False

    logging("Args: " + str(args), args.log)

    logging('loading dataset')
    dataset = pickle.load(open(args.dataset_folder + 'test.pk', 'rb'))
    w_map, test_data, range_idx = dataset['w_map'], dataset['test_data'], dataset['range']

    train_loader = LargeDataset(args.dataset_folder, range_idx, args.batch_size, args.sequence_length)
    test_loader = EvalDataset(test_data, args.batch_size)

    logging('building model')
    lm_model = LM(len(w_map), args.word_dim, args.rnn_unit, args.num_layers, args.hid_dim, dropout=args.dropout, cutoffs=args.cutoffs)
    lm_model.to(device)
    args.device = device

    logging('# of Parameters: %d' % sum([param.numel() for param in lm_model.parameters()]), args.log)

    opt = args.opt
    epochs = args.epochs
    clip = args.clip
    clip_mode = args.clip_mode
    lr_warmup = args.warmup_updates
    init_lr = args.init_lr
    lr_decay = args.lr_decay
    decay_rate = args.decay_rate
    milestone = args.milestone
    rebound = args.rebound
    optimizer, scheduler, opt_param = get_optimizer(opt, args.lr, lm_model.parameters(), warmup_updates=lr_warmup, init_lr=init_lr,
                                                    lr_decay=lr_decay, decay_rate=decay_rate, milestone=milestone, rebound=rebound)
    create_graph = args.opt == 'adahessian'

    if args.recover:
        checkpoint_name = args.checkpoint_name
        print(f"loading from checkpoint {checkpoint_name}")
        checkpoint = torch.load(checkpoint_name, map_location=args.device)
        lm_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch']
        batch_index = checkpoint['batch_index']
        best_epoch = checkpoint['best_epoch']
        best_ppl = checkpoint['best_ppl']
        numbers = checkpoint['numbers']
        train_loss = checkpoint['train_loss']
        del checkpoint
        with torch.no_grad():
            logging('Evaluating after resuming model...', args.log)
            test_ppl = evaluate(args, test_loader, lm_model)
            logging('test_ppl: {} @ epoch: {}, best_ppl: {} @ epoch: {}'.format(test_ppl, start_epoch - 1, best_ppl, best_epoch), args.log)
    else:
        start_epoch = 1
        batch_index = 0
        best_epoch = 0
        best_ppl = float('inf')
        numbers = {'train ppl': [], 'test ppl': []}
        train_loss = 0

    for epoch in range(start_epoch, epochs + 1):
        lr = scheduler.get_last_lr()[0]
        logging('#' * 90, args.log)
        logging('Epoch: {}/{} ({}, lr={:.6f}, clip={:.1f} ({}), {})'.format(epoch, epochs, opt, lr, clip, clip_mode, opt_param), args.log)
        iterator = train_loader.get_tqdm()
        full_epoch_loss = 0
        lm_model.train()

        hx = None
        for word_t, label_t in iterator:
            optimizer.zero_grad()

            if 1 == train_loader.cur_idx:
                hx = None

            word_t = word_t.to(device, non_blocking=True)
            label_t = label_t.to(device, non_blocking=True).view(-1)

            loss, hx = lm_model(word_t, label_t, hx=hx)
            train_loss += loss.item()
            if batch_index > 0 and batch_index % args.interval == 0:
                train_ppl = math.exp(train_loss / args.interval)
                logging('epoch_ppl: {} lr: {} @ batch_index: {}'.format(train_ppl, lr, batch_index), args.log)
                train_loss = 0
                numbers['train ppl'].append(train_ppl)

            batch_index += 1
            loss.backward(create_graph=create_graph)
            if clip > 0.:
                clip_grad_norm_(lm_model.parameters(), clip, mode=clip_mode)
            optimizer.step()

        scheduler.step()

        with torch.no_grad():
            test_ppl = evaluate(args, test_loader, lm_model)
            if test_ppl < best_ppl:
                best_ppl = test_ppl
                best_epoch = epoch
            logging('test_ppl: {} @ epoch: {}, best_ppl: {} @ epoch: {}'.format(test_ppl, epoch, best_ppl, best_epoch), args.log)
            numbers['test ppl'].append(test_ppl)
            json.dump(numbers, open(os.path.join(args.model_path, 'values.run{}.json'.format(args.run)), 'w'))

        # save checkpoint
        checkpoint_name = args.checkpoint_name
        torch.save({'epoch': epoch + 1,
                    'batch_index': batch_index,
                    'model': lm_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_epoch': best_epoch,
                    'best_ppl': best_ppl,
                    'train_loss': train_loss,
                    'numbers': numbers},
                   checkpoint_name)
