# encoding=utf-8
import sys
import random
import numpy as np
import torch
import torch.utils.data as data
from nltk.corpus import stopwords
from itertools import chain
import codecs
import json
import collections
import torch.nn.utils.rnn as rnn_utils


class MyDataset(data.Dataset):
    def __init__(self, corp, config, mode='TRAIN'):
        self.data_convs = []
        self.data_users = []
        self.data_labels = []
        self.pretrain_labels = []
        self.SR_labels = []
        self.AR_labels = []
        self.upattern_labels = []
        self.pretrain = False
        self.pretrain_type = config.pretrain_type
        self.multitask_type = config.multitask_type
        self.multi_task = config.multi_task
        if mode == 'TRAIN':
            convs = corp.convs
            labels = corp.labels
        elif mode == 'TEST':
            convs = corp.test_convs
            labels = corp.test_labels
        else:
            convs = corp.valid_convs
            labels = corp.valid_labels
        pretrain_label_counter_sr = collections.Counter()
        pretrain_label_counter_up = collections.Counter()
        for cid in convs:
            self.data_labels.append(labels[cid])
            self.data_convs.append([turn[1] for turn in convs[cid]])
            self.data_users.append([turn[1] for turn in corp.users[convs[cid][-1][0]][-config.history_size:]])
            if 'SR' in config.pretrain_type or 'SR' in config.multitask_type:
                pretrain_label = 0
                for turn in convs[cid][:-1]:
                    if turn[0] == convs[cid][-1][0]:
                        pretrain_label = 1
                        break
                if config.inverse_labeling:
                    pretrain_label = 1 - pretrain_label
                pretrain_label_counter_sr[pretrain_label] += 1
                self.SR_labels.append(pretrain_label)
            if 'AR' in config.pretrain_type or 'AR' in config.multitask_type:
                pretrain_label = [0 for _ in range(len(convs[cid])-1)]
                for t in range(len(convs[cid])-1):
                    if convs[cid][t][0] == convs[cid][-1][0]:
                        pretrain_label[t] = 1
                if config.inverse_labeling:
                    pretrain_label = [1 - pl for pl in pretrain_label]
                self.AR_labels.append(pretrain_label)
            if 'upattern' in config.pretrain_type or 'upattern' in config.multitask_type:
                users = set()
                for turn in convs[cid]:
                    users.add(turn[0])
                pretrain_label = 1 if len(list(users)) == 2 else 0
                if config.inverse_labeling:
                    pretrain_label = 1 - pretrain_label
                pretrain_label_counter_up[pretrain_label] += 1
                self.upattern_labels.append(pretrain_label)

        if config.pretrain_weight is None:
            if 'SR' in config.pretrain_type or 'SR' in config.multitask_type:
                self.pretrain_weight_sr = float(pretrain_label_counter_sr[0]) / float(pretrain_label_counter_sr[1])
            if 'upattern' in config.pretrain_type or 'upattern' in config.multitask_type:
                self.pretrain_weight_up = float(pretrain_label_counter_up[0]) / float(pretrain_label_counter_up[1])
        else:
            self.pretrain_weight_sr = config.pretrain_weight
            self.pretrain_weight_up = config.pretrain_weight
        if mode == 'TRAIN' and ('SR' in config.pretrain_type or 'SR' in config.multitask_type):
            print('SR Pretrain Labels: ', pretrain_label_counter_sr, 'SR Pretrain Weight: ', self.pretrain_weight_sr)
        if mode == 'TRAIN' and ('upattern' in config.pretrain_type or 'upattern' in config.multitask_type):
            print('upattern Pretrain Labels: ', pretrain_label_counter_up, 'upattern Pretrain Weight: ', self.pretrain_weight_up)

    def __getitem__(self, idx):
        if self.pretrain:
            pretrain_labels = []
            if len(self.SR_labels) != 0:
                pretrain_labels.append(self.SR_labels[idx])
            if len(self.AR_labels) != 0:
                pretrain_labels.append(self.AR_labels[idx])
            if len(self.upattern_labels) != 0:
                pretrain_labels.append(self.upattern_labels[idx])
            return self.data_convs[idx], self.data_users[idx], pretrain_labels
        elif self.multi_task and not self.pretrain:
            pretrain_labels = []
            if len(self.SR_labels) != 0:
                pretrain_labels.append(self.SR_labels[idx])
            if len(self.AR_labels) != 0:
                pretrain_labels.append(self.AR_labels[idx])
            if len(self.upattern_labels) != 0:
                pretrain_labels.append(self.upattern_labels[idx])
            return self.data_convs[idx], self.data_users[idx], self.data_labels[idx], pretrain_labels
        else:
            return self.data_convs[idx], self.data_users[idx], self.data_labels[idx]

    def __len__(self):
        return len(self.data_labels)

    def pad_vector(self, texts, text_size, sent_len):  # Pad with 0s to fixed size
        text_vec = []
        text_len = []
        turn_len = []
        for one_text in texts:
            t = []
            tl = []
            for sent in one_text:
                pad_len = max(0, sent_len - len(sent))
                t.append(sent + [0] * pad_len)
                tl.append(len(sent))
            pad_size = max(0, text_size - len(t))
            text_len.append(len(t))
            t.extend([[0] * sent_len] * pad_size)
            tl.extend([0] * pad_size)
            text_vec.append(t)
            turn_len.append(tl)
        padded_vec = torch.LongTensor(text_vec)
        return padded_vec, text_len, turn_len

    def my_collate(self, batch):
        conv_vecs = [item[0] for item in batch]
        user_vecs = [item[1] for item in batch]
        conv_turn_size = max([len(c) for c in conv_vecs])
        conv_turn_len = max([len(sent) for sent in chain.from_iterable([c for c in conv_vecs])])
        conv_vecs, conv_lens, conv_turn_lens = self.pad_vector(conv_vecs, conv_turn_size, conv_turn_len)
        user_hist_size = max([len(h) for h in user_vecs])
        user_turn_len = max([len(sent) for sent in chain.from_iterable([h for h in user_vecs])])
        user_vecs, user_lens, user_turn_lens = self.pad_vector(user_vecs, user_hist_size, user_turn_len)

        if self.pretrain:
            pretrain_types = self.pretrain_type.split('+')
            pretrain_num = len(pretrain_types)
            pretrain_labels = []
            for i in range(pretrain_num):
                if pretrain_types[i] != 'AR':
                    pretrain_labels.append(torch.Tensor([item[2][i] for item in batch]))
                else:
                    ar_labels = [item[2][i] for item in batch]
                    pretrain_labels.append(torch.cat([torch.Tensor(label) for label in ar_labels]))
            return conv_vecs, conv_lens, conv_turn_lens, user_vecs, user_lens, user_turn_lens, pretrain_labels
        else:
            my_labels = [item[2] for item in batch]
            my_labels = torch.Tensor(my_labels)
            if self.multi_task:
                pretrain_types = self.multitask_type.split('+')
                pretrain_num = len(pretrain_types)
                pre_labels = []
                for i in range(pretrain_num):
                    if pretrain_types[i] != 'AR':
                        pre_labels.append(torch.Tensor([item[3][i] for item in batch]))
                    else:
                        ar_labels = [item[3][i] for item in batch]
                        pre_labels.append(torch.cat([torch.Tensor(label) for label in ar_labels]))
                return conv_vecs, conv_lens, conv_turn_lens, user_vecs, user_lens, user_turn_lens, my_labels, pre_labels
            else:
                return conv_vecs, conv_lens, conv_turn_lens, user_vecs, user_lens, user_turn_lens, my_labels
        

class Corpus:

    def __init__(self, config):
        self.turnNum = 0             # Number of messages
        self.convNum = 0            # Number of conversations
        self.userNum = 0            # Number of users
        self.userIDs = {}           # Dictionary that maps users to integer IDs
        self.r_userIDs = {}         # Inverse of last dictionary
        self.wordNum = 2            # Number of words
        self.wordIDs = {'<Pad>': 0, '<UNK>': 1}           # Dictionary that maps words to integers
        self.r_wordIDs = {0: '<Pad>', 1: '<UNK>'}         # Inverse of last dictionary
        self.wordCount = collections.Counter()            # Record the frequency of each appeared word
        if config.pretrain_type == "IDF_Att" or config.pretrain_type == "TPIDF_Att":
            self.global_word_record = collections.defaultdict(set)
            self.local_word_record = collections.defaultdict(dict)

        # Each conv is a list of turns, each turn is [userID, [w1, w2, w3, ...]]
        self.convs = collections.defaultdict(list)
        self.labels = {}
        # Each user is a list of turns, each turn is [convID, [w1, w2, w3, ...]]
        self.users = collections.defaultdict(list)

        self.test_convs = collections.defaultdict(list)
        self.test_labels = {}
        self.valid_convs = collections.defaultdict(list)
        self.valid_labels = {}

        with codecs.open(config.train_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                msgs = json.loads(line)
                current_turn_num = 0
                for turn in msgs[0]:
                    user_id = turn[2]
                    if user_id not in self.userIDs:
                        self.userIDs[user_id] = self.userNum
                        self.r_userIDs[self.userNum] = user_id
                        self.userNum += 1
                    words = []
                    for word in turn[3].split(' '):
                        if word not in self.wordIDs:
                            self.wordIDs[word] = self.wordNum
                            self.r_wordIDs[self.wordNum] = word
                            self.wordNum += 1
                        if config.pretrain_type == "IDF_Att" or config.pretrain_type == "TPIDF_Att":
                            self.global_word_record[self.wordIDs[word]].add(self.convNum)
                            try:
                                self.local_word_record[self.convNum][self.wordIDs[word]].add(current_turn_num)
                            except KeyError:
                                self.local_word_record[self.convNum][self.wordIDs[word]] = set()
                                self.local_word_record[self.convNum][self.wordIDs[word]].add(current_turn_num)
                        self.wordCount[self.wordIDs[word]] += 1
                        words.append(self.wordIDs[word])
                    self.convs[self.convNum].append([self.userIDs[user_id], words])
                    self.users[self.userIDs[user_id]].append([self.convNum, words])
                    self.turnNum += 1
                    current_turn_num += 1
                self.labels[self.convNum] = msgs[-1]
                self.convNum += 1
        self.train_data = MyDataset(self, config, 'TRAIN')
        print("Corpus initialization over! UserNum: %d ConvNum: %d TurnNum: %d" % (self.userNum, self.convNum, self.turnNum))

    def test_corpus(self, test_file, mode='TEST'):  # mode == 'TEST' or mode == 'VALID'
        with codecs.open(test_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                msgs = json.loads(line)
                for turn in msgs[0]:
                    user_id = turn[2]
                    if user_id not in self.userIDs:
                        self.userIDs[user_id] = self.userNum
                        self.r_userIDs[self.userNum] = user_id
                        self.userNum += 1
                    words = []
                    for word in turn[3].split(' '):
                        try:
                            words.append(self.wordIDs[word])
                        except KeyError:  # for the words that is out of vocabulary
                            words.append(self.wordIDs['<UNK>'])
                            # if word not in self.oovIDs:
                            #     self.oovIDs[word] = len(self.oovIDs)
                            #     self.r_oovIDs[self.oovIDs[word]] = word
                    if len(words) == 0:  # in case some turns are null turn without words
                        words.append(self.wordIDs['<UNK>'])
                    if mode == 'TEST':
                        self.test_convs[self.convNum].append([self.userIDs[user_id], words])
                    else:
                        self.valid_convs[self.convNum].append([self.userIDs[user_id], words])
                if mode == 'TEST':
                    self.test_labels[self.convNum] = msgs[-1]
                else:
                    self.valid_labels[self.convNum] = msgs[-1]
                self.convNum += 1
        print("%s Corpus process over!" % mode)


def create_embedding_matrix(dataname, word_idx, word_num, embedding_dim=200):
    pretrain_file = 'glove.twitter.27B.200d.txt' if dataname[0] == 't' else 'glove.6B.200d.txt'
    pretrain_words = {}
    with open(pretrain_file, 'r') as f:
        for line in f:
            infos = line.split()
            wd = infos[0]
            vec = np.array(infos[1:]).astype(np.float)
            pretrain_words[wd] = vec
    weights_matrix = np.zeros((word_num, embedding_dim))
    for idx in word_idx.keys():
        if idx == 0:
            continue
        try:
            weights_matrix[idx] = pretrain_words[word_idx[idx]]
        except KeyError:
            weights_matrix[idx] = np.random.normal(size=(embedding_dim,))
    if torch.cuda.is_available():  # run in GPU
        return torch.Tensor(weights_matrix).cuda()
    else:
        return torch.Tensor(weights_matrix)


def pretrain_corpus_construction(batch, config):
    convs, conv_lens, conv_turn_lens = batch[0], batch[1], batch[2]
    users, user_lens, user_turn_lens = batch[3], batch[4], batch[5]
    labels = batch[-1]
    if config.pretrain_type == "RR":
        need_replace = torch.rand(len(labels))
        need_replace = need_replace.le(config.Prob)
        for i in range(len(labels)):
            if need_replace[i]:
                if user_lens[i] == 0:
                    turn_len = conv_turn_lens[i][conv_lens[i]-1]
                    convs[i, conv_lens[i]-1, :turn_len] = convs[i, conv_lens[i]-1, torch.randperm(turn_len)]
                else:
                    replace_idx = random.randint(0, user_lens[i]-1)
                    replace_turn_len = min(max(convs.size(-1), conv_turn_lens[i][conv_lens[i]-1]), user_turn_lens[i][replace_idx])
                    convs[i, conv_lens[i]-1, :replace_turn_len] = users[i, replace_idx, :replace_turn_len]
                    conv_turn_lens[i][conv_lens[i] - 1] = replace_turn_len
                labels[i] = 1
            else:
                labels[i] = 0
    elif config.pretrain_type == "REPLACE":
        conv_num = len(convs)
        turn_num = max(conv_lens)
        labels = torch.zeros((conv_num, turn_num))
        for c in range(conv_num):
            for t in range(turn_num):
                if t >= conv_lens[c]:
                    break
                if random.random() <= config.Prob:
                    labels[c, t] = 1
                    rc = random.choice([i for i in range(conv_num) if i != c])
                    rt = random.choice([i for i in range(conv_lens[rc])])
                    convs[c, t, :] = convs[rc, rt, :]
                    conv_turn_lens[c][t] = conv_turn_lens[rc][rt]
        labels = torch.cat([labels[i, : conv_lens[i] - 1] for i in range(convs.size(0))])
    elif config.pretrain_type == "SWITCH":
        conv_num = len(convs)
        turn_num = max(conv_lens)
        labels = torch.zeros((conv_num, turn_num))
        for c in range(conv_num):
            need_switch = [random.random() <= config.Prob for i in range(conv_lens[c])]
            if sum(need_switch) <= 1:
                switch_idx = random.sample(list(range(conv_lens[c])), 2)
            else:
                switch_idx = [i for i in range(conv_lens[c]) if need_switch[i]]
            original_idx = list(switch_idx)
            random.shuffle(switch_idx)
            for i, idx in enumerate(switch_idx):
                if idx == original_idx[i]:
                    switch_idx[i], switch_idx[(i+1) % len(switch_idx)] = switch_idx[(i+1) % len(switch_idx)], switch_idx[i]
            convs[c, original_idx, :] = convs[c, switch_idx, :]
            new_turn_len = [l for l in conv_turn_lens[c]]
            for i in range(len(original_idx)):
                new_turn_len[original_idx[i]] = conv_turn_lens[c][switch_idx[i]]
            conv_turn_lens[c] = new_turn_len
            labels[c, original_idx] = 1
        labels = torch.cat([labels[i, : conv_lens[i] - 1] for i in range(convs.size(0))])
    else:
        print('Wrong Pretrain Type!')
        exit(0)
    return convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens, labels





