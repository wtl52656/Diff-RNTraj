import nni
import time
from tqdm import tqdm
import logging
import sys
import argparse
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
from models.multi_train import evaluate, init_weights, train, generate_data
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
            'id_size':6265+1,
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
    
    if opts.dataset == 'Porto':
        test_flag = True
        # test_flag = False

        path_dir = '/data/WeiTongLong/data/traj_gen/A_new_dataset/Porto_0928/'

        extra_info_dir = path_dir + "extra_file/"
        rn_dir = path_dir + "Porto_OSM/"
        road_file = path_dir + 'graph/graph_A.csv'
        SE_file = path_dir + 'graph/Porto_SE_50.txt'
        if test_flag:
            train_trajs_dir = path_dir + 'gen_debug/'
        else:
            train_trajs_dir = path_dir + 'gen_all/' 
                
    elif opts.dataset == 'Chengdu':
        print("chengdu dataset")
        test_flag = True
        # test_flag = False
        
        path_dir = '/data/WeiTongLong/data/traj_gen/A_new_dataset/chengdu/'

        extra_info_dir = path_dir + "extra_file/"
        rn_dir = path_dir + "road/"
        road_file = path_dir + 'graph/graph_A.csv'
        SE_file = path_dir + 'graph/chengdu_SE.txt'
        if test_flag:
            train_trajs_dir = path_dir + 'gen_debug/'
        else:
            train_trajs_dir = path_dir + 'gen_all/'
                
    model_save_path = './results/'+opts.dataset + '/add_spatial_loss/'
    create_dir(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=model_save_path + 'log.txt',
                        filemode='a')
    # spatial embedding
    spatial_A = load_graph_adj_mtx(road_file)
    spatial_A_trans = np.zeros((spatial_A.shape[0]+1, spatial_A.shape[1]+1)) + 1e-10
    spatial_A_trans[1:,1:] = spatial_A

    f = open(SE_file, mode = 'r')
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

    model_path = './results/{}/add_spatial_loss/'.format(args.dataset)
    model.load_state_dict(torch.load(model_path + 'val-best-model.pt'))
    start_time = time.time()
    generate_data(model, spatial_A_trans, rn_dict, args, SE)