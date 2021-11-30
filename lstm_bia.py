import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import torch.nn.utils.rnn as rnn_utils


class SentLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size, bi_direction=True):
        super(SentLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction
        self.sent_lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bi_direction)
        self.sent_hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        bi = 2 if self.bi_direction else 1
        if torch.cuda.is_available():  # run in GPrU
            return (torch.randn(bi, batch_size, self.hidden_dim).cuda(),
                    torch.randn(bi, batch_size, self.hidden_dim).cuda())
        else:
            return (torch.randn(bi, batch_size, self.hidden_dim),
                    torch.randn(bi, batch_size, self.hidden_dim))

    def forward(self, sentences, sent_lens):
        # print sentences.size()
        self.sent_hidden = self.init_hidden(len(sentences))
        sorted_sent_lens, indices = torch.sort(sent_lens, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_sentences = sentences[indices]
        packed_sentences = rnn_utils.pack_padded_sequence(sorted_sentences, sorted_sent_lens, batch_first=True)
        lstm_out, self.sent_hidden = self.sent_lstm(packed_sentences, self.sent_hidden)
        if self.bi_direction:
            sent_reps = torch.cat([self.sent_hidden[0][-2], self.sent_hidden[0][-1]], dim=1)
            sent_reps = sent_reps[desorted_indices]
        else:
            sent_reps = self.sent_hidden[0][-1][desorted_indices]

        return sent_reps


class LSTMBiA(nn.Module):
    def __init__(self, config):
        super(LSTMBiA, self).__init__()
        self.word_embedding = nn.Embedding(config.vocab_num, config.embedding_dim, padding_idx=0)
        if config.embedding_matrix is not None:
            self.word_embedding.load_state_dict({'weight': config.embedding_matrix})
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim // 2
        self.batch_size = config.batch_size
        self.use_gpu = config.use_gpu
        self.conv_num_layer = 1
        self.model_num_layer = 2
        self.bi_direction = True
        self.dropout = nn.Dropout(config.dropout)
        self.mode = 'train'  # or 'test'

        self.sent_lstm = SentLSTM(self.embedding_dim, self.hidden_dim, self.batch_size)
        self.conv_lstm = nn.LSTM(config.hidden_dim, self.hidden_dim, num_layers=self.conv_num_layer, bidirectional=True)
        self.conv_hidden = self.init_hidden(self.batch_size, self.conv_num_layer)

        self.similarity = nn.Linear(config.hidden_dim * 3, 1)

        self.model_lstm = nn.LSTM(config.hidden_dim, self.hidden_dim, dropout=config.dropout, num_layers=self.model_num_layer, bidirectional=True)
        self.model_hidden = self.init_hidden(self.batch_size, self.model_num_layer)
        self.layer1 = nn.Linear(config.hidden_dim * 4, config.hidden_dim * 2)
        self.layer2 = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.out_layer = nn.Linear(config.hidden_dim, 1)  # Giving the final prediction
        self.final = nn.Sigmoid()

    def init_hidden(self, batch_size, num_layer):
        bi = 2 if self.bi_direction else 1
        if torch.cuda.is_available():  # run in GPU
            return (torch.randn(bi * num_layer, batch_size, self.hidden_dim).cuda(),
                    torch.randn(bi * num_layer, batch_size, self.hidden_dim).cuda())
        else:
            return (torch.randn(bi * num_layer, batch_size, self.hidden_dim),
                    torch.randn(bi * num_layer, batch_size, self.hidden_dim))

    # def forward(self, target_conv, user_history):
    def forward(self, convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens):
        self.conv_hidden = self.init_hidden(len(convs), self.conv_num_layer)
        self.model_hidden = self.init_hidden(len(convs), self.model_num_layer)

        conv_reps = []
        for c in range(len(convs)):
            # turn_num = 0
            # sent_lens = []
            # for turn in conv:
            #     if turn[1] == 0:
            #         break
            #     turn_num += 1
            #     zero_num = torch.sum(turn[1:] == 0)  # find if there are 0s for padding
            #     sent_lens.append(len(turn) - 1 - zero_num)
            # turn_infos = conv[:turn_num, :1].float()
            if torch.cuda.is_available() and self.use_gpu:  # run in GPU
                # turn_infos = turn_infos.cuda()
                sent_reps = self.sent_lstm(self.word_embedding(convs[c, :conv_lens[c]].cuda()), torch.LongTensor(conv_turn_lens[c][:conv_lens[c]]).cuda())
            else:
                sent_reps = self.sent_lstm(self.word_embedding(convs[c, :conv_lens[c]]), torch.LongTensor(conv_turn_lens[c][:conv_lens[c]]))
            conv_reps.append(sent_reps)
        sorted_conv_turn_nums, sorted_conv_indices = torch.sort(torch.LongTensor(conv_lens), descending=True)
        _, desorted_conv_indices = torch.sort(sorted_conv_indices, descending=False)
        sorted_conv_reps = []
        for index in sorted_conv_indices:
            sorted_conv_reps.append(conv_reps[index])
        paded_convs = rnn_utils.pad_sequence(sorted_conv_reps)
        packed_convs = rnn_utils.pack_padded_sequence(paded_convs, sorted_conv_turn_nums)
        conv_out, self.conv_hidden = self.conv_lstm(packed_convs, self.conv_hidden)
        conv_out = rnn_utils.pad_packed_sequence(conv_out, batch_first=True)[0]
        conv_out = conv_out[desorted_conv_indices]

        if self.mode == 'test':
            user_history = []
            for u in range(len(users)):
                current_user = [users[u, i] for i in range(user_lens[u])]
                current_user.append(convs[u, conv_lens[u]-1])
                current_user = rnn_utils.pad_sequence(current_user, batch_first=True)
                user_history.append(current_user)
                user_lens[u] += 1
        else:
            user_history = users
        sorted_his_lens, sorted_his_indices = torch.sort(torch.LongTensor(user_lens), descending=True)
        _, desorted_his_indices = torch.sort(sorted_his_indices, descending=False)
        sorted_user_history = []
        for index in sorted_his_indices:
            sorted_user_history.append(user_history[index])
        his_out = []
        his_num = 0
        for one_his in sorted_user_history:
            sent_lens = []
            for sent in one_his[:sorted_his_lens[his_num]]:
                zero_num = torch.sum(sent == 0)  # find if there are 0s for padding
                sent_lens.append(len(sent) - zero_num)
            if torch.cuda.is_available():  # run in GPU
                his_out.append(self.sent_lstm(self.word_embedding(one_his[:sorted_his_lens[his_num]].cuda()), torch.LongTensor(sent_lens).cuda()))
            else:
                his_out.append(self.sent_lstm(self.word_embedding(one_his[:sorted_his_lens[his_num]]), torch.LongTensor(sent_lens)))
            his_num += 1
        his_out = rnn_utils.pad_sequence(his_out, batch_first=True)
        his_out = his_out[desorted_his_indices]

        batch_size = conv_out.size(0)
        conv_sent_len = conv_out.size(1)
        his_sent_len = his_out.size(1)
        hidden_dim = conv_out.size(-1)

        conv_rep = conv_out.repeat(1, 1, his_sent_len).view(batch_size, conv_sent_len * his_sent_len, -1)
        his_rep = his_out.repeat(1, conv_sent_len, 1)
        sim_matrix = torch.cat([conv_rep, his_rep, conv_rep * his_rep], -1).view(batch_size, conv_sent_len, his_sent_len, -1)
        sim_matrix = self.similarity(sim_matrix)
        sim_matrix = sim_matrix.squeeze(-1)
        # print sim_matrix

        atten_c2h = F.softmax(sim_matrix, dim=-1)
        atten_c2h = atten_c2h.unsqueeze(-1).repeat(1, 1, 1, hidden_dim)
        atten_c2h = atten_c2h * his_out.unsqueeze(1)
        atten_c2h = atten_c2h.sum(2)
        # print F.softmax(sim_matrix, dim=-1)
        # print F.softmax(sim_matrix.max(2)[0], dim=-1)

        atten_h2c = F.softmax(sim_matrix.max(2)[0], dim=-1)
        atten_h2c = atten_h2c.unsqueeze(-1) * conv_out
        atten_h2c = atten_h2c.sum(1)
        atten_h2c = atten_h2c.unsqueeze(1).repeat(1, conv_sent_len, 1)

        conv_rep_atten = torch.cat([conv_out, atten_c2h, conv_out * atten_c2h, conv_out * atten_h2c], dim=-1)
        conv_rep_atten = F.relu(self.layer1(conv_rep_atten))
        conv_rep_atten = F.relu(self.layer2(conv_rep_atten))
        sorted_conv_rep_atten = conv_rep_atten[sorted_conv_indices]
        packed_conv_rep_atten = rnn_utils.pack_padded_sequence(sorted_conv_rep_atten, sorted_conv_turn_nums, batch_first=True)
        _, self.model_hidden = self.model_lstm(packed_conv_rep_atten, self.model_hidden)
        if self.bi_direction:
            model_out = torch.cat([self.model_hidden[0][-2], self.model_hidden[0][-1]], dim=1)
        else:
            model_out = self.model_hidden[0][-1]
        model_out = model_out[desorted_conv_indices]

        conv_labels = self.final(self.out_layer(model_out).view(-1))
        return conv_labels, model_out




