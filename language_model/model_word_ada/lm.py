import torch.nn as nn


class LM(nn.Module):

    def __init__(self, w_num, w_dim, rnn_unit, num_layers, hidden_dim, dropout, cutoffs):
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            w_num: (int): write your description
            w_dim: (int): write your description
            rnn_unit: (int): write your description
            num_layers: (int): write your description
            hidden_dim: (int): write your description
            dropout: (str): write your description
            cutoffs: (float): write your description
        """
        super(LM, self).__init__()

        self.w_num = w_num
        self.w_dim = w_dim
        self.word_embed = nn.Embedding(w_num, w_dim)

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        self.rnn = rnnunit_map[rnn_unit](w_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.soft_max = nn.AdaptiveLogSoftmaxWithLoss(hidden_dim, w_num, cutoffs=cutoffs, div_value=4.0)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the embedding.

        Args:
            self: (todo): write your description
        """
        nn.init.kaiming_uniform_(self.word_embed.weight)

        for param in self.rnn.parameters():
            if param.dim() == 2:
                nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                nn.init.constant_(param, 0.)
                if isinstance(self.rnn, nn.LSTM):
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2 * hidden_size] = 1.0
            else:
                raise ValueError('unexpected parameter for RNN {}'.format(param.size()))

    def detach_hx(self, hx):
        """
        Detach an hxh object.

        Args:
            self: (todo): write your description
            hx: (todo): write your description
        """
        if isinstance(hx, tuple):
            hx, cx = hx
            hx = (hx.detach(), cx.detach())
        else:
            hx = hx.detach()
        return hx

    def forward(self, w_in, target, hx):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            w_in: (todo): write your description
            target: (todo): write your description
            hx: (todo): write your description
        """
        w_emb = self.dropout(self.word_embed(w_in))

        out, hx = self.rnn(w_emb, hx=hx)
        out = self.dropout(out)

        out = out.view(-1, out.size(2))
        out = self.soft_max(out, target)

        return out.loss, self.detach_hx(hx)

    def log_prob(self, w_in, hx):
        """
        Parameters ---------- w_in : int ) ] ) ]

        Args:
            self: (todo): write your description
            w_in: (todo): write your description
            hx: (todo): write your description
        """
        w_emb = self.dropout(self.word_embed(w_in))

        out, hx = self.rnn(w_emb, hx=hx)
        out = self.dropout(out)

        out = out.view(-1, out.size(2))
        out = self.soft_max.log_prob(out)

        hx = self.detach_hx(hx)
        return out, hx
