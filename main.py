import os
import sys
import random
import torch
import time
import argparse
import numpy as np
from data_process import Corpus, MyDataset, create_embedding_matrix
from mymodel import VEP
from lstm_bia import LSTMBiA
from train import train
from predict import predict


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, choices=["test", "twitter", "reddit"])
    parser.add_argument("modelname", type=str, choices=["VEP", "LSTMBIA"])
    parser.add_argument("--cuda_dev", type=str, default="0")
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--history_size", type=int, default=50)
    parser.add_argument("--pre_lr", type=float, default=0.0001)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no_pretrained_embedding", action="store_true")
    parser.add_argument("--train_weight", type=float, default=1)
    parser.add_argument("--pretrain_weight", type=float, default=None)
    parser.add_argument("--no_use_gpu", dest='use_gpu', action='store_false')
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--l2_weight", type=float, default=0.0003)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="0")
    parser.add_argument("--pretrain_path", type=str, default=None)  # None: need pretrain, 'N': don't need pretrain
    parser.add_argument("--no_att", dest='use_att', action='store_false')
    parser.add_argument("--no_history", dest='use_hist', action='store_false')
    parser.add_argument("--hist_concat", action="store_true")
    parser.add_argument("--SR_pretrain", action="store_true")
    parser.add_argument("--AR_pretrain", action="store_true")
    parser.add_argument("--upattern_pretrain", action="store_true")
    parser.add_argument("--SR_multitask", action="store_true")
    parser.add_argument("--AR_multitask", action="store_true")
    parser.add_argument("--upattern_multitask", action="store_true")
    parser.add_argument("--SR_tradeoff", type=float, default=1.0)
    parser.add_argument("--AR_tradeoff", type=float, default=1.0)
    parser.add_argument("--upattern_tradeoff", type=float, default=1.0)
    parser.add_argument("--multitask_tradeoff", type=float, default=0.2)
    parser.add_argument("--att_type", type=str, default='learn', choices=['additive', 'general', 'learn'])
    parser.add_argument("--valid_during_epoch", type=int, default=-1)
    parser.add_argument("--inverse_labeling", action="store_true")

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    config = parse_config()
    setup_seed(config.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_dev
    config.pretrain_type = " "
    config.pretrain_type += "SR+" if config.SR_pretrain else ""
    config.pretrain_type += "AR+" if config.AR_pretrain else ""
    config.pretrain_type += "upattern+" if config.upattern_pretrain else ""
    config.pretrain_type = config.pretrain_type[:-1].strip() if len(config.pretrain_type[:-1].strip()) != 0 else "N"
    config.multitask_type = " "
    config.multitask_type += "SR+" if config.SR_multitask else ""
    config.multitask_type += "AR+" if config.AR_multitask else ""
    config.multitask_type += "upattern+" if config.upattern_multitask else ""
    config.multitask_type = config.multitask_type[:-1].strip() if len(config.multitask_type[:-1].strip()) != 0 else "N"
    if config.multitask_type != "N":
        config.multi_task = True
    else:
        config.multi_task = False
    assert config.pretrain_type == config.multitask_type or config.pretrain_type == "N" or config.multitask_type == "N"
    print('Start processing! File name: %s. Model name: %s, Pretrain Type: %s, Multitask Tyoe: %s.' % (config.filename, config.modelname, config.pretrain_type, config.multitask_type))

    if config.filename == "test":
        config.train_file, config.test_file, config.valid_file = "t_train_"+config.data_type+".json", "t_test_"+config.data_type+".json", "t_valid_"+config.data_type+".json"
    elif config.filename == "twitter":
        config.train_file, config.test_file, config.valid_file = "twitter_train_"+config.data_type+".json", "twitter_test_"+config.data_type+".json", "twitter_valid_"+config.data_type+".json"
    elif config.filename == "reddit":
        config.train_file, config.test_file, config.valid_file = "reddit_train_"+config.data_type+".json", "reddit_test_"+config.data_type+".json", "reddit_valid_"+config.data_type+".json"
    else:
        print('Data name not correct!')
        sys.exit()
    corp = Corpus(config)
    config.vocab_num = corp.wordNum
    if config.no_pretrained_embedding or config.filename == "test" or config.mode == "predict":
        config.embedding_matrix = None
    else:
        config.embedding_matrix = create_embedding_matrix(config.filename, corp.r_wordIDs, corp.wordNum, config.embedding_dim)
    config.path = "Results/" + config.modelname + "/" + config.filename + "/"
    if not os.path.isdir(config.path):
        os.makedirs(config.path)
    if config.modelname == "VEP":
        model = VEP(config)
        config.path += config.pretrain_type.replace('+', '-') + "_" + config.multitask_type.replace('+', '-') + "_"
    else:
        model = LSTMBiA(config)
    print('Parameter Size in Main: ', sum(p.numel() for name, p in model.named_parameters() if 'ar' not in name and 'sr' not in name and 'up' not in name))
    print('Parameter Size in Total: ', sum(p.numel() for p in model.parameters()))
    config.loss_weights = torch.Tensor([1, config.train_weight])
    if torch.cuda.is_available() and config.use_gpu:  # run in GPU
        model = model.cuda()
        config.loss_weights = config.loss_weights.cuda()
    if config.mode == "train":
        train(corp, model, config)
    else:
        predict(corp, model, config)



