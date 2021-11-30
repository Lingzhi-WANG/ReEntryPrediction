import sys
import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import math
import torch.nn.utils.rnn as rnn_utils


class TurnEncoder(nn.Module):
    def __init__(self, config, input_dim, hidden_dim):
        super(TurnEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_gpu = config.use_gpu
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, dropout=config.dropout, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, sentences, sentence_lengths, initial_vectors=None):
        sorted_sentence_lengths, indices = torch.sort(sentence_lengths, descending=True)
        sorted_sentences = sentences[indices]
        _, desorted_indices = torch.sort(indices, descending=False)
        packed_sentences = rnn_utils.pack_padded_sequence(sorted_sentences, sorted_sentence_lengths, batch_first=True)
        if initial_vectors is not None:
            initial_vectors[0] = initial_vectors[0][indices]
            initial_vectors[1] = initial_vectors[1][indices]
            _, output = self.gru(packed_sentences, initial_vectors)
        else:
            _, output = self.gru(packed_sentences)
        output = torch.cat([output[-1], output[-2]], dim=-1)[desorted_indices]
        output = self.dropout(output)
        return output


class VEP(nn.Module):
    def __init__(self, config):
        super(VEP, self).__init__()
        self.word_embedding = nn.Embedding(config.vocab_num, config.embedding_dim, padding_idx=0)
        if config.embedding_matrix is not None:
            self.word_embedding.load_state_dict({'weight': config.embedding_matrix})
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.batch_size = config.batch_size
        self.use_gpu = config.use_gpu
        self.use_att = config.use_att
        self.use_hist = config.use_hist
        self.use_hist_concat = config.hist_concat
        self.multi_task = config.multi_task
        self.dropout = nn.Dropout(config.dropout)
        self.pretrain_type = config.pretrain_type.split('+')
        self.multitask_type = config.multitask_type.split('+')
        self.att_type = config.att_type
        self.mode = 'train'  # or 'pretrain'

        self.turn_encoder = TurnEncoder(config, config.embedding_dim, config.hidden_dim)
        self.conv_encoder = nn.GRU(self.hidden_dim*2, self.hidden_dim, dropout=config.dropout, bidirectional=True)
        if self.use_hist:
            self.user_encoder = nn.GRU(self.hidden_dim*2, self.hidden_dim, dropout=config.dropout, bidirectional=True)
            self.hidden2init = nn.Linear(self.hidden_dim*2, self.embedding_dim)

        if self.use_att:
            if self.att_type == 'additive':
                self.att_w = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2, bias=False)
                self.att_v = nn.Linear(self.hidden_dim * 2, 1, bias=False)
            elif self.att_type == 'general':
                self.att = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2, bias=False)
            else:
                self.att = nn.Linear(self.hidden_dim*2, 1)
            self.hidden2label_ar = nn.Linear(self.hidden_dim * 2, 1)
            if self.use_hist_concat:
                self.hidden2label_sr = nn.Linear(self.hidden_dim * 6, 1)
                self.hidden2label_up = nn.Linear(self.hidden_dim * 6, 1)
                self.hidden2label_final = nn.Linear(self.hidden_dim * 6, 1)
            else:
                self.hidden2label_sr = nn.Linear(self.hidden_dim * 4, 1)
                self.hidden2label_up = nn.Linear(self.hidden_dim * 4, 1)
                self.hidden2label_final = nn.Linear(self.hidden_dim * 4, 1)

        else:
            self.hidden2label_ar = nn.Linear(self.hidden_dim * 2, 1)
            if self.use_hist_concat:
                self.hidden2label_sr = nn.Linear(self.hidden_dim * 4, 1)
                self.hidden2label_up = nn.Linear(self.hidden_dim * 4, 1)
                self.hidden2label_final = nn.Linear(self.hidden_dim * 4, 1)
            else:
                self.hidden2label_sr = nn.Linear(self.hidden_dim * 2, 1)
                self.hidden2label_up = nn.Linear(self.hidden_dim * 2, 1)
                self.hidden2label_final = nn.Linear(self.hidden_dim*2, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def init_hidden(self, batch_size, hidden_dim, zero_init=False):
        if torch.cuda.is_available() and self.use_gpu:  # run in GPU
            if zero_init:
                return torch.zeros(2, batch_size, hidden_dim).cuda()
            else:
                return torch.randn(2, batch_size, hidden_dim).cuda()
        else:
            if zero_init:
                return torch.zeros(2, batch_size, hidden_dim)
            else:
                return torch.randn(2, batch_size, hidden_dim)

    def forward(self, convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens):
        """convs: conversation input tokens [batch_size, conv_len, token_num]
           conv_lens: lengths of each conversation [batch_size]
           conv_turn_lens: lengths of each turn in conversation [batch_size, conv_len]
           users: user history input tokens [batch_size, history_len, token_num]
           user_lens: lengths of each user history [batch_size]
           user_turn_lens: lengths of each turn in user history [batch_size, history_len]"""

        """User history processing"""
        if self.use_hist:
            sorted_user_lens, sorted_user_indices = torch.sort(torch.LongTensor(user_lens), descending=True)
            _, desorted_user_indices = torch.sort(sorted_user_indices, descending=False)
            user_reps = []
            i = 0
            while i < len(sorted_user_indices):
                # Encode each turn in user history
                idx = sorted_user_indices[i]
                current_user = users[idx]
                if user_lens[idx] == 0:  # the rest users do not have history
                    break
                else:
                    current_reps = self.turn_encoder(self.word_embedding(current_user[:user_lens[idx]]), torch.LongTensor(user_turn_lens[idx][:user_lens[idx]]))
                user_reps.append(current_reps)
                i += 1
            if i > 0:
                padded_users = rnn_utils.pad_sequence(user_reps, batch_first=True)
                packed_padded_users = rnn_utils.pack_padded_sequence(padded_users, sorted_user_lens[:i], batch_first=True)
                user_initial_hidden = self.init_hidden(len(padded_users), self.hidden_dim)
                # print(padded_users.size(), sorted_user_lens[:i].size(), user_initial_hidden.size())
                # Encode the whole user history
                _, user_output = self.user_encoder(packed_padded_users, user_initial_hidden)
                user_reps = torch.cat([user_output[-1], user_output[-2]], dim=-1)  # Each user's reps
            else:
                user_reps = torch.Tensor([])
            if i != len(user_lens):
                # randomly initialize those users who do not have history
                if torch.cuda.is_available() and self.use_gpu:  # run in GPU
                    user_reps = torch.cat([user_reps, torch.randn((len(user_lens)-i, self.hidden_dim*2)).cuda()], dim=0)
                else:
                    user_reps = torch.cat([user_reps, torch.randn((len(user_lens)-i, self.hidden_dim*2))], dim=0)
            user_reps = user_reps[desorted_user_indices]

        """Conversation processing"""
        sorted_conv_lens, sorted_conv_indices = torch.sort(torch.LongTensor(conv_lens), descending=True)
        _, desorted_conv_indices = torch.sort(sorted_conv_indices, descending=False)
        conv_reps = []
        for idx in sorted_conv_indices:
            current_conv = convs[idx]
            current_len = conv_lens[idx]
            # Encode each turn in conversation, target turn initialized with user reps
            turn_init = self.init_hidden(current_len, self.embedding_dim, zero_init=True)
            if self.use_hist and not self.use_hist_concat:
                turn_init[0, current_len-1] = turn_init[1, current_len-1] = self.tanh(self.hidden2init(user_reps[idx]))
            context_reps = self.turn_encoder(self.word_embedding(current_conv[:current_len]), torch.LongTensor(conv_turn_lens[idx][:current_len]), turn_init)
            conv_reps.append(context_reps)
        padded_convs = rnn_utils.pad_sequence(conv_reps, batch_first=True)
        packed_padded_convs = rnn_utils.pack_padded_sequence(padded_convs, sorted_conv_lens, batch_first=True)
        conv_initial_hidden = self.init_hidden(len(padded_convs), self.hidden_dim)
        # Encode the whole conversation
        conv_output, conv_hidden = self.conv_encoder(packed_padded_convs, conv_initial_hidden)
        conv_output = rnn_utils.pad_packed_sequence(conv_output, batch_first=True)[0][desorted_conv_indices]
        target_turn = torch.cat([conv_hidden[-1], conv_hidden[-2]], dim=1)[desorted_conv_indices]

        """Attention Module"""
        if self.use_att:
            if torch.cuda.is_available() and self.use_gpu:  # run in GPU
                masks = torch.where(conv_output.sum(dim=-1) != 0, torch.Tensor([0.]).cuda(), torch.Tensor([-np.inf]).cuda())
            else:
                masks = torch.where(conv_output.sum(dim=-1) != 0, torch.Tensor([0.]), torch.Tensor([-np.inf]))
            if self.att_type == 'additive':
                for i in range(len(conv_lens)):
                    masks[i, conv_lens[i] - 1] = -np.inf
                att_scores = torch.cat([conv_output, target_turn.unsqueeze(1).repeat(1, conv_output.size(1), 1)], dim=2)
                att_scores = self.att_v(self.tanh(self.att_w(att_scores))).squeeze(-1)
            elif self.att_type == 'general':
                for i in range(len(conv_lens)):
                    masks[i, conv_lens[i] - 1] = -np.inf
                att_scores = self.att(conv_output)
                att_scores = (att_scores * target_turn.unsqueeze(1)).sum(dim=-1)
            else:
                att_scores = self.att(conv_output).squeeze(-1)
            att_scores = F.softmax(att_scores + masks - torch.max(att_scores, dim=1, keepdim=True)[0], dim=1)
            turn_scores = torch.mul(conv_output, att_scores.unsqueeze(2))
            att_out = torch.bmm(conv_output.transpose(1, 2), att_scores.unsqueeze(2)).squeeze(2)
            final_out = torch.cat([target_turn, att_out], dim=1)

        else:
            att_scores = None
            turn_scores = conv_output
            final_out = target_turn
        if self.use_hist_concat:
            final_out = torch.cat([final_out, user_reps], dim=1)


        """Prediction Layer"""
        if self.mode == 'pretrain' or self.multi_task:
            pre_labels = []
            p_type = self.pretrain_type if self.mode == 'pretrain' else self.multitask_type
            for i, pt in enumerate(p_type):
                if pt == 'SR':
                    pre_labels.append(self.sigmoid(self.dropout(self.hidden2label_sr(final_out)).view(-1)))
                elif pt == 'AR':
                    if self.att_type in ['additive', 'general']:
                        ar_labels = self.sigmoid(self.dropout(self.hidden2label_ar(turn_scores)).squeeze(-1))
                    else:
                        ar_labels = self.sigmoid((turn_scores * target_turn.unsqueeze(1)).sum(dim=-1))
                    pre_labels.append(torch.cat([ar_labels[i, : conv_lens[i]-1] for i in range(turn_scores.size(0))]))
                elif pt == 'upattern':
                    pre_labels.append(self.sigmoid(self.dropout(self.hidden2label_up(final_out)).view(-1)))
        if self.mode == 'pretrain':
            return pre_labels, att_scores
        conv_labels = self.sigmoid(self.dropout(self.hidden2label_final(final_out)).view(-1))
        if self.multi_task:
            return (conv_labels, pre_labels), att_scores
        else:
            return conv_labels, att_scores

        












