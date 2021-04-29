import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Char_CNN(nn.Module):
    def __init__(self, args, config):

        super().__init__()
        cnn_info = config.model[args.model]["CNN"]
        high_info = config.model[args.model]["High"]
        lstm_info = config.model[args.model]["LSTM"]
        data_info = config.data_info[args.dataset]

        self.mode = args.model
        self.dimension = cnn_info["dimension"]
        self.filter_list, self.size = self.make_filter(cnn_info["window_sizes"], cnn_info["n_filters"])
        self.high_n = high_info["layer"]
        self.lstm_h_units = lstm_info["h_units"]
        self.lstm_n = lstm_info["layer"]

        self.word_vocab = data_info["word_vocab"]
        self.char_vocab = data_info["char_vocab"]
        self.p = max(cnn_info["window_sizes"])

        self.use_batch_norm = args.use_batch_norm
        self.dropout_p = args.dropout_p

        self.bs = config.train.bs
        self.bptt = config.train.bptt

        #stack layers
        self.emb = nn.Embedding(self.char_vocab, self.dimension, max_norm=0.05)
        self.feature_extractors = nn.ModuleList()


        for window_size, n_filter in self.filter_list:
            self.feature_extractors.append(
                nn.Sequential(nn.Conv2d(in_channels=1,
                                        out_channels=n_filter,
                                        kernel_size= (window_size, self.dimension), #(1, 15)
                                        stride= 1, ## 한번에 볼 dimension 이랑 크기 같이 넣어줘야 함
                                        #padding=(self.p, 0),
                                        bias=True),
                              nn.Tanh()))

        self.highway = Highway(self.size, self.high_n)
        self.lstm = nn.LSTM(self.size, self.lstm_h_units, num_layers=self.lstm_n,dropout=self.dropout_p, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(), nn.Linear(self.lstm_h_units, self.word_vocab))


    def forward(self, x): # x:(batch_size=20, sequence_length=35, max_word_length=21)
        x = torch.flatten(x, start_dim=0, end_dim=1) # x : ( batch_size * sequence_length, max_word_length)
        x = self.emb(x)  # x:(batch_size * sequence_length, max_word_length, char_embedding_dimension)
        x = x.unsqueeze(1) # x : (batch_size*sequence_length, 1, max_word_length, char_embedding_dimension)
        cnn_outs = []
        for block in self.feature_extractors:
            cnn_out = block(x)
            cnn_out = nn.functional.max_pool1d(
                input= cnn_out.squeeze(-1),
                kernel_size = cnn_out.size(-2)
            ).squeeze(-1)
            cnn_outs += [cnn_out]
        outs = torch.cat(cnn_outs, dim=-1)


        outs = self.highway.forward(outs) # outs : (bs*bptt, filter_num) (700, 525)
        outs = outs.reshape(self.bs, self.bptt, -1)
        outs,_ = self.lstm(outs) # outs:  20, 35, 300
        outs = self.fc(outs) # 20, 35, 10000
        outs = torch.flatten(outs, start_dim=0, end_dim=1)
        return outs

    def make_filter(self, window_sizes=None, filter_info=None):
        if self.mode == "s":
            filter_list = [filter_info*i for i in window_sizes]

        else: # large model
            n = filter_info[0]
            p = filter_info[1]
            filter_list = [min(n, p*i) for i in window_sizes]
        return list(zip(window_sizes, filter_list)), sum(filter_list)

class Highway(nn.Module):
    def __init__(self, size, num_layers):
        # num_layers -> config.model[s][High][layer] -> 1 or 2
        # size : total filternum -> 525
        self.num_layers = num_layers
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.relu(self.nonlinear[layer](x))

            x = torch.mul(gate, nonlinear) + torch.mul((1 - gate), x)
        return x


