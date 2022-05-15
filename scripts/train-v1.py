import argparse
# from cgi import test
import math
import sys
import os

sys.path.append(os.getcwd())
import pickle
import logging
import random
import time
import socket

import utils
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from numpy import linalg as LA
from torch import autograd
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset
# from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from data import DataLoaderX, TrajectoryDataset, TrajectoryDataset_Real
from metrics import *
# from models.model0 import social_stgcnn0
from testmodel_v1 import test, test_main
# from testmodel import test, test_main
from utils import (
    displacement_error,
    final_displacement_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    relative_to_abs,
)

from models.register import import_all_modules_for_register
from models.register import Registers

# import_all_modules_for_register()

parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument("--model",
                    default="socail_stgcnn",
                    type=str,
                    help="the name of the model")
parser.add_argument("--name", type=str, default="stgcnn", help="dir name")
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn',
                    type=int,
                    default=1,
                    help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn',
                    type=int,
                    default=5,
                    help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset',
                    default='eth',
                    help='eth,hotel,univ,zara1,zara2')

#Training specifc parameters
parser.add_argument('--batch_size',
                    type=int,
                    default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs',
                    type=int,
                    default=250,
                    help='number of epochs')
parser.add_argument('--clip_grad',
                    type=float,
                    default=None,
                    help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_sh_rate',
                    type=int,
                    default=150,
                    help='number of steps to drop the lr')
parser.add_argument('--use_lrschd',
                    action="store_true",
                    default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag',
                    default='',
                    type=str,
                    help='personal tag for the model ')
parser.add_argument("--log_dir",
                    type=str,
                    default="runs",
                    help="the dir of the checkpoint")
parser.add_argument("--seed", default=0, type=int, help='random seed')
parser.add_argument("--gpu_num", default=None, type=str)
parser.add_argument("--drop", type=float, default=0.1, help="dropout rate")
parser.add_argument("--init", default="kmeans", type=str, help="init type")
parser.add_argument("--dict_kernel_size", type=int, default=3, help="kernel size of fuse")
parser.add_argument("--act", type=str, default="nn.PReLU", help="activation function")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
test_dir = os.path.join('./runs', args.name)
log_dir = os.path.join(test_dir, args.tag)
args.log_dir = test_dir
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
utils.set_logger(os.path.join(log_dir, "train.log"))
checkpoint_dir = os.path.join(log_dir, "checkpoint")

for k, v in sorted(vars(args).items()):
    logging.info("{}={}".format(k, v))

logging.info('*' * 30)
logging.info("Training initiating....")
logging.info(args)


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)


if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(checkpoint_dir + '/args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

if args.gpu_num is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

writer = SummaryWriter(log_dir)
global metrics, loader_train

#Data prep
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
data_set = './datasets/' + args.dataset + '/'
logging.info("Initializing train dataset")

dset_train = TrajectoryDataset_Real(data_set + 'train/',
                                    obs_len=obs_seq_len,
                                    pred_len=pred_seq_len,
                                    skip=1,
                                    norm_lap_matr=True)

loader_train = DataLoaderX(
    dset_train,
    batch_size=1,  #This is irrelative to the args batch size parameter
    shuffle=True,
    num_workers=0)

logging.info("Initializing val dataset")
dset_val = TrajectoryDataset_Real(data_set + 'val/',
                                  obs_len=obs_seq_len,
                                  pred_len=pred_seq_len,
                                  skip=1,
                                  norm_lap_matr=True)

loader_val = DataLoaderX(
    dset_val,
    batch_size=1,  #This is irrelative to the args batch size parameter
    shuffle=False,
    num_workers=1)

dset_test = TrajectoryDataset_Real(data_set + 'test/',
                                   obs_len=obs_seq_len,
                                   pred_len=pred_seq_len,
                                   skip=1,
                                   norm_lap_matr=True)

loader_test = DataLoaderX(
    dset_test,
    batch_size=1,  #This is irrelative to the args batch size parameter
    shuffle=False,
    num_workers=1)
# logging.info(len(loader_train))
# logging.info(len(loader_val))
# logging.info(len(loader_test))
#Defining the model
# logging.info(Registers.model.keys())
model = Registers.model[args.model](n_stgcnn=args.n_stgcnn,
                                    n_txpcnn=args.n_txpcnn,
                                    output_feat=args.output_size,
                                    seq_len=args.obs_seq_len,
                                    kernel_size=args.kernel_size,
                                    pred_seq_len=args.pred_seq_len,
                                    drop=args.drop,
                                    init=args.init,
                                    dict_kernel_size=args.dict_kernel_size,
                                    act=args.act,
                                    env=args.tag).cuda()
logging.info(model.__class__.__name__)

#Training settings

optimizer = optim.SGD(model.parameters(), lr=args.lr)
# scaler = GradScaler()
if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.lr_sh_rate,
                                          gamma=0.2)

logging.info('Data and model loaded')
logging.info('Checkpoint dir: {}'.format(str(checkpoint_dir)))

#Training
metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}


def train(epoch):
    losses = utils.AverageMeter("Loss", ":.4f")
    batch_time = utils.AverageMeter("Time: ", ":.3f")
    data_time = utils.AverageMeter("Data Time: ", ":.3f")
    progress = utils.ProgressMeter(len(loader_train),
                                   [losses, batch_time, data_time],
                                   prefix="Train: [{}]".format(epoch))

    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    # print(loader_len)
    turn_point = int(loader_len / args.batch_size
                     ) * args.batch_size + loader_len % args.batch_size - 1
    flag = time.time()

    for cnt, batch in enumerate(loader_train):
        batch_count += 1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
        loss_mask,V_obs,A_obs,V_tr,A_tr = batch
        data_time.update(time.time() - flag)
        # print(obs_traj.size(), pred_traj_gt.size())
        # print(V_obs.size())

        optimizer.zero_grad()
        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        # print(V_obs_tmp.size())

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), obs_traj)
        # print(V_tr.size(), V_pred.size())

        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            losses.update(loss.item(), obs_traj.shape[1])
            # scaler.scale(loss)
            loss.backward()

            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            #Metrics
            loss_batch += loss.item()
            # logging.info('TRAIN:', '\t Epoch:', epoch, '\t Loss:',
            #       loss_batch / batch_count)
        batch_time.update(time.time() - flag)
        flag = time.time()
    progress.display(cnt)
    remain_epoch = args.num_epochs - epoch
    eta_seconds = batch_time.avg * remain_epoch * loader_len
    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
    logging.info("  Eta: {}".format(eta))
    # prof.step()
    writer.add_scalar("train_loss", losses.avg, epoch)

    # metrics['train_loss'].append(loss_batch / batch_count)


def valid(epoch):
    flag = False
    global metrics, loader_val, constant_metrics
    ade = utils.AverageMeter("ADE", ":.6f")
    fde = utils.AverageMeter("FDE", ":.6f")
    losses = utils.AverageMeter("Loss", ":.4f")
    progress = utils.ProgressMeter(len(loader_val), [losses],
                                   prefix="Valid: [{}]".format(epoch))

    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size
                     ) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1

        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), obs_traj)

        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            #Metrics
            loss_batch += loss.item()
            losses.update(loss_batch)
            # logging.info('Valid:', '\t Epoch:', epoch, '\t Loss:',
            #       loss_batch / batch_count)
    progress.display(cnt)
    writer.add_scalar("valid_loss", losses.avg, epoch)
    metrics['val_loss'].append(loss_batch / batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + '/val_best.pth')  # OK
        logging.info("Checkpoint saved...")
        flag = True
    
    return flag


logging.info('Training started ...')
for epoch in range(args.num_epochs):
    train(epoch)
    if epoch > 150:
        flag = valid(epoch)
        # if flag is True:
        #     test(loader_test, model, writer=writer, epoch=epoch)
    if args.use_lrschd:
        scheduler.step()

    with open(checkpoint_dir + '/metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(checkpoint_dir + '/constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)

if args.tag == "zara2":
    test_main(args)
# os.system("python ./testmodel.py --log_dir {} --model {}".format(
# test_dir, args.model))
writer.close()
