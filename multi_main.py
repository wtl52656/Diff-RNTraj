import nni
import time
from tqdm import tqdm
import logging
import sys
import argparse
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.optim as optim
import numpy as np
from utils.utils import save_json_data, create_dir, load_pkl_data
from common.mbr import MBR
from common.spatial_func import SPoint, distance
from common.road_network import load_rn_shp
from torch.optim.lr_scheduler import StepLR


from utils.datasets import Dataset, collate_fn, LoadData
from models.model_utils import load_rn_dict, load_rid_freqs, get_rid_grid, get_poi_info, get_rn_info
from models.model_utils import get_online_info_dict, epoch_time, AttrDict, get_rid_rnfea_dict
from models.multi_train import init_weights, train
from models.model import Diff_RNTraj
from models.diff_module import diff_CSDI
from build_graph import load_graph_adj_mtx, load_graph_node_features
import warnings
import json
import pickle
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Multi-task Traj Interp')
    parser.add_argument('--dataset', type=str, default='Chengdu',help='data set')
    parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--diff_T', type=int, default=500, help='diffusion step')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='min beta')
    parser.add_argument('--beta_end', type=float, default=0.02, help='max beta')
    parser.add_argument('--pre_trained_dim', type=int, default=64, help='pre-trained dim of the road segment')
    parser.add_argument('--rdcl', type=int, default=10, help='stack layers on the denoise network')
    
    
    opts = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = AttrDict()

    if opts.dataset == 'Porto':
        args_dict = {
            'dataset': opts.dataset,
            # MBR
            'min_lat':41.142,
            'min_lng':-8.652,
            'max_lat':41.174,
            'max_lng':-8.578,
            'grid_size': 50, 

            # model params
            'hid_dim':opts.hid_dim,
            'id_size':13695+1,
            'n_epochs':opts.epochs,
            'batch_size':opts.batch_size,
            'learning_rate':opts.lr,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1,

            'diff_T': opts.diff_T,
            'beta_start': opts.beta_start,
            'beta_end': opts.beta_end,
            'pre_trained_dim': opts.pre_trained_dim,
            'rdcl': opts.rdcl
        }
    elif opts.dataset == 'Chengdu':
        args_dict = {
            'dataset': opts.dataset,

            # MBR
            'min_lat':30.655,
            'min_lng':104.043,
            'max_lat':30.727,
            'max_lng':104.129,
            'grid_size': 50, 

            # model params
            'hid_dim':opts.hid_dim,
            'id_size':6256+1,
            'n_epochs':opts.epochs,
            'batch_size':opts.batch_size,
            'learning_rate':opts.lr,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1,

            'diff_T': opts.diff_T,
            'beta_start': opts.beta_start,
            'beta_end': opts.beta_end,
            'pre_trained_dim': opts.pre_trained_dim,
            'rdcl': opts.rdcl
        }
    
    assert opts.dataset in ['Porto', 'Chengdu'], 'Check dataset name if in [Porto, Chengdu]'

    args.update(args_dict)

    print('Preparing data...')
    
    beta = np.linspace(opts.beta_start ** 0.5, opts.beta_end ** 0.5, opts.diff_T) ** 2
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    alpha = torch.tensor(alpha).float().to("cuda:0")
    alpha_bar = torch.tensor(alpha_bar).float().to("cuda:0")

    diffusion_hyperparams = {}
    diffusion_hyperparams['T'], diffusion_hyperparams['alpha_bar'], diffusion_hyperparams['alpha'] = opts.diff_T,  alpha_bar, alpha
    diffusion_hyperparams['beta'] = beta
    
    test_flag = True
    # test_flag = False

    if opts.dataset == 'Porto':
        path_dir = '/data/WeiTongLong/data/traj_gen/A_new_dataset/Porto/'
    elif opts.dataset == 'Chengdu':
        path_dir = '/data/WeiTongLong/data/traj_gen/A_new_dataset/Chengdu/'


    extra_info_dir = path_dir + "extra_file/"
    rn_dir = path_dir + "road_network/"
    UTG_file = path_dir + 'graph/graph_A.csv'
    pre_trained_road = path_dir + 'graph/road_embed.txt'
    if test_flag:
        train_trajs_dir = path_dir + 'gen_debug/'
    else:
        train_trajs_dir = path_dir + 'gen_all/' 
                
                
    model_save_path = './results/'+opts.dataset + '/'
    create_dir(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=model_save_path + 'log.txt',
                        filemode='a')
    # spatial embedding
    spatial_A = load_graph_adj_mtx(UTG_file)
    spatial_A_trans = np.zeros((spatial_A.shape[0]+1, spatial_A.shape[1]+1)) + 1e-10
    spatial_A_trans[1:,1:] = spatial_A

    f = open(pre_trained_road, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0])+1, int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index+1] = temp[1 :]
    
    SE = torch.from_numpy(SE)

    rn = load_rn_shp(rn_dir, is_directed=True)
    raw_rn_dict = load_rn_dict(extra_info_dir, file_name='raw_rn_dict.json')
    new2raw_rid_dict = load_rid_freqs(extra_info_dir, file_name='new2raw_rid.json')
    raw2new_rid_dict = load_rid_freqs(extra_info_dir, file_name='raw2new_rid.json')
    rn_dict = load_rn_dict(extra_info_dir, file_name='rn_dict.json')

    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    grid_rn_dict, max_xid, max_yid = get_rid_grid(mbr, args.grid_size, rn_dict)
    args_dict['max_xid'] = max_xid
    args_dict['max_yid'] = max_yid
    args.update(args_dict)
    print(args)
    logging.info(args_dict)
    with open(model_save_path+'logging.txt', 'w') as f:
        f.write(str(args_dict))
        f.write('\n')
        

    # load dataset
    with open(train_trajs_dir + 'eid_seqs.bin', 'rb') as f:  #路段序列
        all_src_eid_seqs = pickle.load(f)
        f.close()
    # with open()
    with open(train_trajs_dir + 'rate_seqs.bin', 'rb') as f:  #路段序列
        all_src_rate_seqs = pickle.load(f)
        f.close()

    diff_model = diff_CSDI(args.hid_dim, args.hid_dim, opts.diff_T, args.hid_dim, args.pre_trained_dim, args.rdcl)
    model = Diff_RNTraj(diff_model, diffusion_hyperparams).to(device)
    model.apply(init_weights)  # learn how to init weights
    
    print('model', str(model))
    logging.info('model' + str(model))
    with open(model_save_path+'logging.txt', 'a+') as f:
        f.write('model' + str(model) + '\n')
        
    ls_train_loss, ls_train_const_loss, ls_train_diff_loss, ls_train_x0_loss = [], [], [], []
    
    dict_train_loss = {}
    
    best_loss = float('inf')  # compare id loss

    # get all parameters (model parameters + task dependent log variances)
    log_vars = [torch.zeros((1,), requires_grad=True, device=device)] * 2  # use for auto-tune multi-task param
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    scheduler = StepLR(optimizer, 
                step_size = 3, # Period of learning rate decay
                gamma = 0.5)
    for epoch in tqdm(range(args.n_epochs)):
        start_time = time.time()

        new_log_vars, train_loss, train_const_loss, train_diff_loss, train_x0_loss = \
                train(model, spatial_A_trans, SE, all_src_eid_seqs, all_src_rate_seqs, optimizer, log_vars, args, diffusion_hyperparams)
        scheduler.step()
        

        ls_train_loss.append(train_loss)
        ls_train_const_loss.append(train_const_loss)
        ls_train_diff_loss.append(train_diff_loss)
        ls_train_x0_loss.append(train_x0_loss)

        dict_train_loss['train_ttl_loss'] = ls_train_loss
        dict_train_loss['train_const_loss'] = ls_train_const_loss
        dict_train_loss['train_diff_loss'] = ls_train_diff_loss
        dict_train_loss['train_x0_loss'] = ls_train_x0_loss

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if train_loss < best_loss:
            best_loss = train_loss
            print("saveing.....")
            torch.save(model.state_dict(), model_save_path + 'val-best-model.pt')

        logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
        weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
        logging.info('log_vars:' + str(weights))
        logging.info('\tTrain Loss:' + str(train_loss) +
                        '\tTrain Const Loss:' + str(train_const_loss) +
                        '\tTrain Diff Loss:' + str(train_diff_loss) +
                        '\tTrain X0 Loss:' + str(train_x0_loss))
        with open(model_save_path+'logging.txt', 'a+') as f:
            f.write('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's' + '\n')
            f.write('\tTrain Loss:' + str(train_loss) +
                        '\tTrain Const Loss:' + str(train_const_loss) +
                        '\tTrain Diff Loss:' + str(train_diff_loss) +
                        '\tTrain X0 Loss:' + str(train_x0_loss) +
                        '\n')
        torch.save(model.state_dict(), model_save_path + 'train-mid-model.pt')
        save_json_data(dict_train_loss, model_save_path, "train_loss.json")
                
