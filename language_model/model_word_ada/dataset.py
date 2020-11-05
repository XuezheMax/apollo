import sys
import pickle
from tqdm import tqdm
import numpy as np

import torch


class EvalDataset(object):

    def __init__(self, dataset, sequence_length):
        """
        Initialize dataset.

        Args:
            self: (todo): write your description
            dataset: (todo): write your description
            sequence_length: (int): write your description
        """
        super(EvalDataset, self).__init__()
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.construct_index()

    def get_tqdm(self):
        """
        Return the tqdm interval

        Args:
            self: (todo): write your description
        """
        return tqdm(self, mininterval=2, total=self.index_length, leave=False, file=sys.stdout, ncols=80)

    def construct_index(self):
        """
        Construct a tensor.

        Args:
            self: (todo): write your description
        """
        token_per_batch = self.sequence_length
        tot_num = len(self.dataset) - 1
        res_num = tot_num - tot_num % token_per_batch

        self.x = list(torch.unbind(torch.LongTensor(self.dataset[0:res_num]).view(-1, self.sequence_length), 0))
        self.y = list(torch.unbind(torch.LongTensor(self.dataset[1:res_num + 1]).view(-1, self.sequence_length), 0))

        self.x.append(torch.LongTensor(self.dataset[res_num:tot_num]))
        self.y.append(torch.LongTensor(self.dataset[res_num + 1:tot_num + 1]))

        self.index_length = len(self.x)
        self.cur_idx = 0

    def __iter__(self):
        """
        Returns an iterator over the iterable.

        Args:
            self: (todo): write your description
        """
        return self

    def __next__(self):
        """
        Return the next word.

        Args:
            self: (todo): write your description
        """
        if self.cur_idx == self.index_length:
            self.cur_idx = 0
            raise StopIteration

        word_t = self.x[self.cur_idx].unsqueeze(1)
        label_t = self.y[self.cur_idx].unsqueeze(1)

        self.cur_idx += 1

        return word_t, label_t


class LargeDataset(object):

    def __init__(self, root, range_idx, batch_size, sequence_length):
        """
        Initialize the batch.

        Args:
            self: (todo): write your description
            root: (str): write your description
            range_idx: (str): write your description
            batch_size: (int): write your description
            sequence_length: (int): write your description
        """
        super(LargeDataset, self).__init__()
        self.root = root
        self.range_idx = range_idx
        self.shuffle_list = list(range(0, range_idx))

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.token_per_batch = self.batch_size * self.sequence_length

        self.total_batch_num = -1
        self.batch_count = 0

    def shuffle(self):
        """
        Shuffle the list of the elements.

        Args:
            self: (todo): write your description
        """
        np.random.shuffle(self.shuffle_list)

    def get_tqdm(self):
        """
        Return a tqdm object.

        Args:
            self: (todo): write your description
        """
        self.batch_count = 0

        if self.total_batch_num <= 0:
            return tqdm(self, mininterval=2, leave=False, file=sys.stdout).__iter__()
        else:
            return tqdm(self, mininterval=2, total=self.total_batch_num, leave=False, file=sys.stdout, ncols=80).__iter__()

    def __iter__(self):
        """
        Iterate iterator over the file.

        Args:
            self: (todo): write your description
        """
        self.cur_idx = 0
        self.file_idx = 0
        self.index_length = 0
        self.shuffle()
        return self

    def __next__(self):
        """
        Return the next word.

        Args:
            self: (todo): write your description
        """
        if self.cur_idx >= self.index_length:
            self.open_next()

        word_t = self.x[self.cur_idx]
        label_t = self.y[self.cur_idx]
        self.cur_idx += 1
        return word_t, label_t

    def open_next(self):
        """
        Open the next batch.

        Args:
            self: (todo): write your description
        """
        if self.file_idx >= self.range_idx:
            self.total_batch_num = self.batch_count
            raise StopIteration

        lines = pickle.load(open(self.root + 'train_' + str(self.shuffle_list[self.file_idx]) + '.pk', 'rb'))
        np.random.shuffle(lines)
        dataset = np.concatenate(lines)

        res_num = len(dataset) - 1
        res_num = res_num - res_num % self.token_per_batch

        self.x = torch.LongTensor(dataset[0:res_num]).view(self.batch_size, -1, self.sequence_length)
        self.x.transpose_(0, 1).transpose_(1, 2)
        self.y = torch.LongTensor(dataset[1:res_num + 1]).view(self.batch_size, -1, self.sequence_length)
        self.y.transpose_(0, 1).transpose_(1, 2)

        self.index_length = self.x.size(0)
        self.cur_idx = 0

        self.batch_count += self.index_length
        self.file_idx += 1
