import argparse
from email.policy import default
import os
import math
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import *
from data import TrajectoryDataset_Real, TrajectoryDataset
from metrics import *
# from models.model0 import social_stgcnn0
import copy
# from models.register import import_all_modules_for_register
from models.register import Registers

import logging

# import_all_modules_for_register()
# DOMAINS = ["eth"]
DOMAINS = ["eth", "hotel", "univ", "zara1", "zara2"]


def test(loader_test, model, writer=None, epoch=0, KSTEPS=20):
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]

        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), obs_traj)
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0, 2, 3, 1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
        #print(V_pred.shape)

        #For now I have my bi-variate parameters
        #normx =  V_pred[:,:,0:1]
        #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:, :, 2])  #sx
        sy = torch.exp(V_pred[:, :, 3])  #sy
        corr = torch.tanh(V_pred[:, :, 4])  #corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)

        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len

        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(
            V_obs.data.cpu().numpy().squeeze().copy(), V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(
            V_tr.data.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                V_pred.data.cpu().numpy().squeeze().copy(),
                V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(
                copy.deepcopy(V_pred_rel_to_abs))

            # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    if writer is not None:
        writer.add_scalar("test ade", ade_, epoch)
        writer.add_scalar("test fde", fde_, epoch)
    logging.info("Test: [{}]  ADE: {:.4f}  FDE: {:.4f}".format(
        epoch, ade_, fde_))
    return ade_, fde_, raw_data_dict


def test_main(args):
    if args.gpu_num is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    log_dir = args.log_dir
    # model_name = args.model
    # log_dir = [os.path.join(log_dir, name) for name in DOMAINS]
    KSTEPS = 20

    print("*" * 50)
    print('Number of samples:', KSTEPS)
    print("*" * 50)
    ade_ls = []
    fde_ls = []
    # logging.info(log_dir)
    for tag in DOMAINS:
        path = os.path.join(log_dir, tag)
        # exps = glob.glob(path)
        # print('Model being tested are:', exps)
        print(path)
        exp_path = os.path.join(path, "checkpoint")

        # for exp_path in exps:
        # print("*" * 50)
        print("Evaluating model:", exp_path)

        model_path = exp_path + '/val_best.pth'
        args_path = exp_path + '/args.pkl'
        with open(args_path, 'rb') as f:
            args = pickle.load(f)

        stats = exp_path + '/constant_metrics.pkl'
        with open(stats, 'rb') as f:
            cm = pickle.load(f)
        print("Stats:", cm)

        #Data prep
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        data_set = './datasets/' + args.dataset + '/'
        # print(data_set)

        dset_test = TrajectoryDataset_Real(data_set + 'test/',
                                           obs_len=obs_seq_len,
                                           pred_len=pred_seq_len,
                                           skip=1,
                                           norm_lap_matr=True)

        loader_test = DataLoader(
            dset_test,
            batch_size=1,  #This is irrelative to the args batch size parameter
            shuffle=False,
            num_workers=1)

        #Defining the model
        # print(args)
        # print(args.model)
        # print(Registers.model)
        # model = Registers.model[args.model](
        model = Registers.model[args.model](
            n_stgcnn=args.n_stgcnn,
            n_txpcnn=args.n_txpcnn,
            output_feat=args.output_size,
            seq_len=args.obs_seq_len,
            kernel_size=args.kernel_size,
            pred_seq_len=args.pred_seq_len,
            drop=0.0,
            init=args.init,
            dict_kernel_size=args.dict_kernel_size,
            act=args.act,
            # ).cuda()
            env=args.tag).cuda()
        model.eval()
        # drop=args.drop).cuda()

        model.load_state_dict(torch.load(model_path))

        ade_ = 999999
        fde_ = 999999
        print("Testing {} ....".format(tag))
        ad, fd, raw_data_dic_ = test(loader_test, model)
        ade_ = min(ade_, ad)
        fde_ = min(fde_, fd)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        print("ADE: ", ade_, " FDE: ", fde_)

        print("*" * 50)

    print("Avg ADE: ", sum(ade_ls) / 5)
    print("Avg FDE: ", sum(fde_ls) / 5)

    outputlog = args.name
    ade_log = args.name
    fde_log = args.name
    for i in range(5):
        outputlog += " {:.3f}/{:.3f}".format(ade_ls[i], fde_ls[i])
        ade_log += " {:.3f} ".format(ade_ls[i])
        fde_log += " {:.3f} ".format(fde_ls[i])

    outputlog += "{:.3f}/{:.3f}".format(sum(ade_ls) / 5, sum(fde_ls) / 5)
    ade_log += "{:.3f}".format(sum(ade_ls)/5)
    fde_log += "{:.3f}".format(sum(fde_ls)/5)

    logging.info(outputlog)
    logging.info(ade_log)
    print(ade_log)
    logging.info(fde_log)
    print(fde_log)
    # print(outputlog)


    df = pd.read_csv("./results_ade.csv", index_col=0)
    df_a = pd.DataFrame(
        {
            "name":  [args.name],
            "eth":   [ade_ls[0]],
            "hotel": [ade_ls[1]],
            "univ":  [ade_ls[2]],
            "zara1": [ade_ls[3]],
            "zara2": [ade_ls[4]],
            "avg":   [sum(ade_ls)/5]
        }
    )
    df = pd.concat([df, df_a])

    df.to_csv("./results_ade.csv", index=False)

    df = pd.read_csv("./results_fde.csv", index_col=0)
    df_a = pd.DataFrame(
        {
            "name":  [args.name],
            "eth":   [fde_ls[0]],
            "hotel": [fde_ls[1]],
            "univ":  [fde_ls[2]],
            "zara1": [fde_ls[3]],
            "zara2": [fde_ls[4]],
            "avg":   [sum(fde_ls)/5]
        }
    )
    df = pd.concat([df, df_a])

    df.to_csv("./results_fde.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",
                        type=str,
                        default="runs",
                        help="the dir of the checkpoint")
    # parser.add_argument("--model",
    #                     type=str,
    #                     default="social_stgcnn",
    #                     help="model name")
    # parser.add_argument("--drop", type=float, help="dropout")
    parser.add_argument("--gpu_num", default=None, type=str)
    args = parser.parse_args()
    test_main(args)