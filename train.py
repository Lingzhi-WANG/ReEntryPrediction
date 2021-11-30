import os
import sys
import random
import torch
import math
import time
import argparse
import collections
import numpy as np
from torch import nn, optim
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
from itertools import chain
from data_process import Corpus, MyDataset, pretrain_corpus_construction
from evaluate import weighted_binary_cross_entropy, valid_evaluate, test_evaluate


def pretrain_epoch(model, train_data, loss_weights, optimizer, epoch, config, out_text):
    start = time.time()
    model.train()
    print('Pretrain Epoch: %d start!' % epoch)
    out_text += ('Pretrain Epoch: %d start!\n' % epoch)
    avg_loss = 0.0
    train_loader = data.DataLoader(train_data, collate_fn=train_data.my_collate, batch_size=config.batch_size, num_workers=0, shuffle=True)
    pretrain_types = config.pretrain_type.split('+')
    for batch_idx, batch in enumerate(train_loader):
        convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens, labels = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            convs = convs.cuda()
            users = users.cuda()
            labels = [label.cuda() for label in labels]
        predictions, _ = model(convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens)
        for i, pt in enumerate(pretrain_types):
            if pt == 'SR':
                cur_loss = config.SR_tradeoff * weighted_binary_cross_entropy(predictions[i], labels[i], loss_weights[0])
            elif pt == 'AR':
                cur_loss = config.AR_tradeoff * nn.MSELoss()(predictions[i], labels[i])
            elif pt == 'upattern':
                cur_loss = config.upattern_tradeoff * weighted_binary_cross_entropy(predictions[i], labels[i], loss_weights[-1])
            if i == 0:
                loss = cur_loss
            else:
                loss += cur_loss
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    end = time.time()
    print('Pretrain Epoch: %d done! Train avg_loss: %g! Using time: %.2f minutes!' % (epoch, avg_loss, (end - start) / 60))
    out_text += ('Pretrain Epoch: %d done! Train avg_loss: %g! Using time: %.2f minutes!\n' % (epoch, avg_loss, (end - start) / 60))
    return avg_loss, out_text


def train_epoch(model, train_data, valid_loader, loss_weights, optimizer, epoch, config, out_text, valid_out):
    start = time.time()
    model.train()
    print('Train Epoch: %d start!' % epoch)
    out_text += ('Train Epoch: %d start!\n' % epoch)
    avg_loss = 0.0
    pretrain_types = config.multitask_type.split('+')
    train_loader = data.DataLoader(train_data, collate_fn=train_data.my_collate, batch_size=config.batch_size, num_workers=0, shuffle=True)
    for batch_idx, batch in enumerate(train_loader):
        if config.multi_task:
            convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens, labels, pre_labels = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]
        else:
            convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens, labels = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            convs = convs.cuda()
            users = users.cuda()
            labels = labels.cuda()
            if config.multi_task:
                pre_labels = [pre_label.cuda() for pre_label in pre_labels]
        predictions, _ = model(convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens)
        if config.multi_task:
            loss = weighted_binary_cross_entropy(predictions[0], labels, loss_weights[0])
            for i, pt in enumerate(pretrain_types):
                if pt == 'SR':
                    loss += config.multitask_tradeoff * config.SR_tradeoff * weighted_binary_cross_entropy(predictions[1][i], pre_labels[i], loss_weights[1][0])
                elif pt == 'AR':
                    loss += config.multitask_tradeoff * config.AR_tradeoff * nn.MSELoss()(predictions[1][i], pre_labels[i])
                elif pt == 'upattern':
                    loss += config.multitask_tradeoff * config.upattern_tradeoff * weighted_binary_cross_entropy(predictions[1][i], pre_labels[i], loss_weights[1][-1])
            
        else:
            loss = weighted_binary_cross_entropy(predictions, labels, loss_weights)
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx != 0 and config.valid_during_epoch != -1 and batch_idx % config.valid_during_epoch == 0:
            _, valid_f1, valid_loss, _ = valid_evaluate(model, valid_loader, config)
            model.train()
            valid_out += ('%g\t%g\n' % (valid_f1, valid_loss))
            print('Valid during epoch: %g\t%g' % (valid_f1, valid_loss))
    avg_loss /= len(train_data)
    end = time.time()
    print('Train Epoch: %d done! Train avg_loss: %g! Using time: %.2f minutes!' % (epoch, avg_loss, (end - start) / 60))
    out_text += ('Train Epoch: %d done! Train avg_loss: %g! Using time: %.2f minutes!\n' % (epoch, avg_loss, (end - start) / 60))
    return avg_loss, out_text, valid_out





def train(corp, model, config):
    pretrain_optimizer = optim.Adam(model.parameters(), lr=config.pre_lr, weight_decay=config.l2_weight)
    train_optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2_weight)
    train_data = corp.train_data
    corp.test_corpus(config.valid_file, mode='VALID')
    valid_data = MyDataset(corp, config, 'VALID')
    out_text = ""

    # First step: Pretrain the model
    if config.pretrain_path == "N" or config.pretrain_type == "N" or config.modelname == "LSTMBIA":
        pass
    elif config.pretrain_path is None:
        model.mode = 'pretrain'
        train_data.pretrain = True
        valid_data.pretrain = True
        valid_loader = data.DataLoader(valid_data, collate_fn=valid_data.my_collate, batch_size=config.batch_size, num_workers=0)
        if "SR" in config.pretrain_type and "upattern" in config.pretrain_type:
            config.pretrain_loss_weights = [torch.Tensor([1, train_data.pretrain_weight_sr]), torch.Tensor([1, train_data.pretrain_weight_up])]
        elif "SR" in config.pretrain_type:
            config.pretrain_loss_weights = [torch.Tensor([1, train_data.pretrain_weight_sr])]
        elif "upattern" in config.pretrain_type:
            config.pretrain_loss_weights = [torch.Tensor([1, train_data.pretrain_weight_up])]
        else:
            config.pretrain_loss_weights = None
        if torch.cuda.is_available() and config.use_gpu and config.pretrain_loss_weights is not None:  # run in GPU
            if "SR" in config.pretrain_type and "upattern" in config.pretrain_type:
                config.pretrain_loss_weights = [config.pretrain_loss_weights[0].cuda(), config.pretrain_loss_weights[1].cuda()]
            else:
                config.pretrain_loss_weights = [config.pretrain_loss_weights[0].cuda()]
        best_state = None
        best_valid_f1 = -1.0
        best_valid_loss = 999999.99
        no_improve = 0
        for epoch in range(config.max_epoch):
            _, out_text = pretrain_epoch(model, train_data, config.pretrain_loss_weights, pretrain_optimizer, epoch, config, out_text)
            valid_auc, valid_f1, valid_loss, _ = valid_evaluate(model, valid_loader, config)
            if best_valid_f1 < valid_f1 or best_valid_loss > valid_loss:
                no_improve = 0
                best_state = model.state_dict()
                if best_valid_f1 < valid_f1:
                    best_valid_f1 = valid_f1
                    print('New Best F1 Valid Result!!! Valid F1: %g, Valid Loss: %g' % (valid_f1, valid_loss))
                    out_text += ('New Best F1 Valid Result!!! Valid F1: %g, Valid Loss: %g\n' % (valid_f1, valid_loss))
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    print('New Best Loss Valid Result!!! Valid F1: %g, Valid Loss: %g' % (valid_f1, valid_loss))
                    out_text += ('New Best Loss Valid Result!!! Valid F1: %g, Valid Loss: %g\n' % (valid_f1, valid_loss))
            else:
                no_improve += 1
                print('No improve! Current Valid F1: %g, Best Valid F1: %g;  Current Valid Loss: %g, Best Valid Loss: %g' % (valid_f1, best_valid_f1, valid_loss, best_valid_loss))
                out_text += ('No improve! Current Valid F1: %g, Best Valid F1: %g;  Current Valid Loss: %g, Best Valid Loss: %g\n' % (valid_f1, best_valid_f1, valid_loss, best_valid_loss))
            if no_improve == 2:
                break
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(torch.load(config.pretrain_path))

    # Second step: Fine-tune the model
    model.mode = 'train'
    valid_out = "F1\tLoss\n"
    train_data.pretrain = False
    valid_data.pretrain = False
    valid_loader = data.DataLoader(valid_data, collate_fn=valid_data.my_collate, batch_size=config.batch_size, num_workers=0)
    if config.pretrain_type != "IDF_Att" and config.pretrain_type != "TPIDF_Att":
        best_state_f1 = None
        best_state_loss = None
        best_valid_thr_f1 = 0.0
        best_valid_thr_loss = 0.0
        best_valid_f1 = -1.0
        best_valid_loss = 999999.99
        no_improve = 0
        for epoch in range(config.max_epoch):
            if config.modelname == 'LSTMBIA':
                model.mode = 'train'
            if config.multi_task:
                if "SR" in config.multitask_type and "upattern" in config.multitask_type:
                    config.pretrain_loss_weights = [torch.Tensor([1, train_data.pretrain_weight_sr]),
                                                    torch.Tensor([1, train_data.pretrain_weight_up])]
                elif "SR" in config.multitask_type:
                    config.pretrain_loss_weights = [torch.Tensor([1, train_data.pretrain_weight_sr])]
                elif "upattern" in config.multitask_type:
                    config.pretrain_loss_weights = [torch.Tensor([1, train_data.pretrain_weight_up])]
                else:
                    config.pretrain_loss_weights = None
                if torch.cuda.is_available() and config.use_gpu and config.pretrain_loss_weights is not None:  # run in GPU
                    if "SR" in config.multitask_type and "upattern" in config.multitask_type:
                        config.pretrain_loss_weights = [config.pretrain_loss_weights[0].cuda(),
                                                        config.pretrain_loss_weights[1].cuda()]
                    else:
                        config.pretrain_loss_weights = [config.pretrain_loss_weights[0].cuda()]
                _, out_text, valid_out = train_epoch(model, train_data, valid_loader, (config.loss_weights, config.pretrain_loss_weights), train_optimizer, epoch, config, out_text, valid_out)
            else:
                _, out_text, valid_out = train_epoch(model, train_data, valid_loader, config.loss_weights, train_optimizer, epoch, config, out_text, valid_out)
            if config.modelname == 'LSTMBIA':
                model.mode = 'test'
            valid_auc, valid_f1, valid_loss, valid_thr = valid_evaluate(model, valid_loader, config)
            if best_valid_f1 < valid_f1 or best_valid_loss > valid_loss:
                no_improve = 0
                if best_valid_f1 < valid_f1:
                    best_state_f1 = model.state_dict()
                    best_valid_thr_f1 = valid_thr
                    best_valid_f1 = valid_f1
                    print('New Best F1 Valid Result!!! Valid F1: %g, Valid Loss: %g' % (valid_f1, valid_loss))
                    out_text += ('New Best F1 Valid Result!!! Valid F1: %g, Valid Loss: %g\n' % (valid_f1, valid_loss))
                if best_valid_loss > valid_loss:
                    best_state_loss = model.state_dict()
                    best_valid_thr_loss = valid_thr
                    best_valid_loss = valid_loss
                    print('New Best Loss Valid Result!!! Valid F1: %g, Valid Loss: %g' % (valid_f1, valid_loss))
                    out_text += ('New Best Loss Valid Result!!! Valid F1: %g, Valid Loss: %g\n' % (valid_f1, valid_loss))
            else:
                no_improve += 1
                print('No improve! Current Valid F1: %g, Best Valid F1: %g;  Current Valid Loss: %g, Best Valid Loss: %g' % (valid_f1, best_valid_f1, valid_loss, best_valid_loss))
                out_text += ('No improve! Current Valid F1: %g, Best Valid F1: %g;  Current Valid Loss: %g, Best Valid Loss: %g\n' % (valid_f1, best_valid_f1, valid_loss, best_valid_loss))
            if no_improve == 5:
                break

    

    # Final step: Evaluate the model
    corp.test_corpus(config.test_file, mode='TEST')
    test_data = MyDataset(corp, config, 'TEST')
    test_loader = data.DataLoader(test_data, collate_fn=test_data.my_collate, batch_size=config.batch_size, num_workers=0)
    if config.modelname == 'LSTMBIA':
        model.mode = 'test'
    model.load_state_dict(best_state_f1)
    res_f1 = test_evaluate(model, test_loader, config, best_valid_thr_f1)
    print('Result in test set(F1 Valid): AUC %g, F1 Score %g, Precision %g, Recall %g, Accuracy %g' % (res_f1[0], res_f1[1], res_f1[2], res_f1[3], res_f1[4]))
    out_text += ('Result in test set(F1 Valid): AUC %g, F1 Score %g, Precision %g, Recall %g, Accuracy %g\n' % (res_f1[0], res_f1[1], res_f1[2], res_f1[3], res_f1[4]))
    model.load_state_dict(best_state_loss)
    res_loss = test_evaluate(model, test_loader, config, best_valid_thr_loss)
    print('Result in test set(Loss Valid): AUC %g, F1 Score %g, Precision %g, Recall %g, Accuracy %g' % (res_loss[0], res_loss[1], res_loss[2], res_loss[3], res_loss[4]))
    out_text += ('Result in test set(Loss Valid): AUC %g, F1 Score %g, Precision %g, Recall %g, Accuracy %g\n' % (res_loss[0], res_loss[1], res_loss[2], res_loss[3], res_loss[4]))
    if res_f1[1] >= res_loss[1]:
        torch.save(best_state_f1, config.path + '%.4f_%.4f_%.4f_%g_%d_%d_%g_f1.model' % (res_f1[0], res_f1[1], best_valid_f1, config.lr, epoch, config.random_seed, config.train_weight))
        with open(config.path + '%.4f_%.4f_%.4f_%g_%d_%d_%g_f1.res' % (res_f1[0], res_f1[1], best_valid_f1, config.lr, epoch, config.random_seed, config.train_weight), 'w') as f:
            f.write('AUC\tF1-Score\tPrecision\tRecall\tAccuracy\n')
            f.write('%g\t%g\t%g\t%g\t%g\n\n' % (res_f1[0], res_f1[1], res_f1[2], res_f1[3], res_f1[4]))
            f.write('Threshold: %g\n' % best_valid_thr_f1)
            f.write('\n\nParameters:\n')
            for key in config.__dict__:
                f.write('%s : %s\n' % (key, config.__dict__[key]))
        with open(config.path + '%.4f_%.4f_%.4f_%g_%d_%d_%g_f1.out' % (res_f1[0], res_f1[1], best_valid_f1, config.lr, epoch, config.random_seed, config.train_weight), 'w') as f:
            f.write(out_text)
        if config.valid_during_epoch != -1:
            with open(config.path + '%.4f_%.4f_%.4f_%g_%d_%d_%g_f1-valid.out' % (res_f1[0], res_f1[1], best_valid_f1, config.lr, epoch, config.random_seed, config.train_weight), 'w') as f:
                f.write(valid_out)
    else:
        torch.save(best_state_loss, config.path + '%.4f_%.4f_%.4f_%g_%d_%d_%g_loss.model' % (res_loss[0], res_loss[1], best_valid_loss, config.lr, epoch, config.random_seed, config.train_weight))
        with open(config.path + '%.4f_%.4f_%.4f_%g_%d_%d_%g_loss.res' % (res_loss[0], res_loss[1], best_valid_loss, config.lr, epoch, config.random_seed, config.train_weight), 'w') as f:
            f.write('AUC\tF1-Score\tPrecision\tRecall\tAccuracy\n')
            f.write('%g\t%g\t%g\t%g\t%g\n\n' % (res_loss[0], res_loss[1], res_loss[2], res_loss[3], res_loss[4]))
            f.write('Threshold: %g\n' % best_valid_thr_loss)
            f.write('\n\nParameters:\n')
            for key in config.__dict__:
                f.write('%s : %s\n' % (key, config.__dict__[key]))
        with open(config.path + '%.4f_%.4f_%.4f_%g_%d_%d_%g_loss.out' % (res_loss[0], res_loss[1], best_valid_loss, config.lr, epoch, config.random_seed, config.train_weight), 'w') as f:
            f.write(out_text)
        if config.valid_during_epoch != -1:
            with open(config.path + '%.4f_%.4f_%.4f_%g_%d_%d_%g_loss-valid.out' % (res_loss[0], res_loss[1], best_valid_loss, config.lr, epoch, config.random_seed, config.train_weight), 'w') as f:
                f.write(valid_out)










