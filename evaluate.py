import os
import sys
import random
import torch
from torch import nn
import math
import time
import argparse
import numpy as np
from data_process import pretrain_corpus_construction
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(torch.clamp(output, min=1e-10, max=1))) + \
            weights[0] * ((1 - target) * torch.log(torch.clamp(1 - output, min=1e-10, max=1)))
    else:
        loss = target * torch.log(torch.clamp(output, min=1e-10, max=1)) + \
               (1 - target) * torch.log(torch.clamp(1 - output, min=1e-10, max=1))

    return torch.neg(torch.mean(loss))


def valid_evaluate(model, valid_loader, config):  # validation, report the valid auc, loss and threshold
    model.eval()
    true_labels = []
    pred_labels = []
    avg_loss = 0.0
    pretrain_types = config.pretrain_type.split('+') if 'pretrain' in model.mode else config.multitask_type.split('+')
    for batch_idx, batch in enumerate(valid_loader):
        # if model.mode == 'train' or model.mode == 'test' or config.pretrain_type in ['SR', 'AR', 'SR+AR', 'upattern']:
        convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens, labels = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        # else:
        #     convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens, labels = pretrain_corpus_construction(batch, config)
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            convs = convs.cuda()
            users = users.cuda()
        predictions, _ = model(convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens)
        # if 'pretrain' in model.mode and config.pretrain_type == "SR+AR":
        if 'pretrain' in model.mode:
            if torch.cuda.is_available() and config.use_gpu:  # run in GPU
                predictions = [pred.cpu() for pred in predictions]
            for i, pt in enumerate(pretrain_types):
                if pt == 'SR':
                    avg_loss += config.SR_tradeoff * weighted_binary_cross_entropy(predictions[i], labels[i]).item()
                elif pt == 'AR':
                    avg_loss += config.AR_tradeoff * nn.MSELoss()(predictions[i], labels[i]).item()
                else:
                    avg_loss += config.upattern_tradeoff * weighted_binary_cross_entropy(predictions[i], labels[i]).item()
            # avg_loss += weighted_binary_cross_entropy(predictions[0], labels[0]).item() + config.pretrain_tradeoff * mse_loss(predictions[1], labels[1]).item()
        if len(pretrain_types) == 1 or model.mode != 'pretrain':
            if type(predictions) is list or type(predictions) is tuple:
                predictions = predictions[0]
            if type(labels) is list or type(labels) is tuple:
                labels = labels[0]
            if torch.cuda.is_available() and config.use_gpu:  # run in GPU
                predictions = predictions.cpu()
            if model.mode == 'pretrain' and config.pretrain_type == 'AR':
                avg_loss += nn.MSELoss()(predictions, labels).item()
                # cur_true_labels = np.concatenate([x for x in labels.data.numpy()])
                # cur_pred_labels = np.concatenate([x for x in predictions.data.numpy()])
                true_labels = np.concatenate([true_labels, labels.data.numpy()])
                pred_labels = np.concatenate([pred_labels, predictions.data.numpy()])
            else:
                avg_loss += weighted_binary_cross_entropy(predictions, labels).item()
                true_labels = np.concatenate([true_labels, labels.data.numpy()])
                pred_labels = np.concatenate([pred_labels, predictions.data.numpy()])
    avg_loss /= len(valid_loader)
    if 'pretrain' in model.mode and len(pretrain_types) != 1:
        return -1, -1, avg_loss, -1
    try:
        auc = roc_auc_score(true_labels, pred_labels)
    except ValueError:
        auc = 0.0

    cur_thr = 0.001
    best_thr = 0.0
    best_fc = 0.0
    while cur_thr < 1:
        current_pred_labels = (pred_labels >= cur_thr)
        fc = f1_score(true_labels, current_pred_labels)
        if fc > best_fc:
            best_fc = fc
            best_thr = cur_thr
        cur_thr += 0.001

    return auc, best_fc, avg_loss, best_thr


def test_evaluate(model, test_loader, config, threshold):  # evaluation, report the auc, f1, pre, rec, acc
    model.eval()
    true_labels = []
    pred_labels = []
    for batch_idx, batch in enumerate(test_loader):
        convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens, labels = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            convs = convs.cuda()
            users = users.cuda()
        predictions, _ = model(convs, conv_lens, conv_turn_lens, users, user_lens, user_turn_lens)
        if type(predictions) is list or type(predictions) is tuple:
            predictions = predictions[0]
        if torch.cuda.is_available() and config.use_gpu:  # run in GPU
            predictions = predictions.cpu()
        true_labels = np.concatenate([true_labels, labels.data.numpy()])
        pred_labels = np.concatenate([pred_labels, predictions.data.numpy()])

    try:
        auc = roc_auc_score(true_labels, pred_labels)
    except ValueError:
        auc = 0.0
    pred_labels = (pred_labels >= threshold)
    acc = accuracy_score(true_labels, pred_labels)
    pre = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)
    f1 = (0 if pre == rec == 0 else 2 * pre * rec / (pre + rec))

    return auc, f1, pre, rec, acc


