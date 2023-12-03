import random
from tqdm import tqdm
import os
from chinese_calendar import is_holiday

import numpy as np
import torch

from common.spatial_func import distance
from common.trajectory import get_tid, Trajectory
from utils.parse_traj import ParseMMTraj
from utils.save_traj import SaveTraj2MM
from utils.utils import create_dir
from .model_utils import load_rid_freqs

EARTH_REDIUS = 6378.137
def rad(d):
    return d * np.pi / 180.0

def getDistance(lat1, lng1, lat2, lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    type_in = np.sqrt(np.sin(a/2)**2 + np.cos(radLat1) * np.cos(radLat2) * np.sin(b/2) ** 2)
    s = 2 * np.arcsin(type_in)
    s = s * EARTH_REDIUS * 1000
    return s


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


def build_graph(src_len, src_grids, src_gps):
    import time
    start = time.time()
    
    batchsize, max_src_length, _ = src_grids.shape
    new_G_time = torch.zeros((batchsize, max_src_length, max_src_length))
    new_G_dist = torch.zeros((batchsize, max_src_length, max_src_length))
    for i in range(max_src_length):
        for j in range(max_src_length):
            new_G_time[:,i,j] = abs(i-j)
            
            ori_lat, ori_lng = src_gps[:,i,0], src_gps[:,i,1]
            dest_lat, dest_lng = src_gps[:,j,0], src_gps[:,j,1]
            
            new_G_dist[:,i,j] = getDistance(ori_lat, ori_lng, dest_lat, dest_lng)
            if i == j:
                new_G_time[:,i,j] = 1e9
                new_G_dist[:,i,j] = 1e6

    new_G_time = torch.exp(-new_G_time)
    new_G_dist[new_G_dist>1e5] = 0  #由于轨迹长度不足max_src_length，最前面的经纬度为0，目的消除经纬度为0的点
    for i in range(batchsize):
        # mask_num = max_src_length - src_len[i]
        new_G_time[i,src_len[i]:] = 0
        new_G_time[i, :, src_len[i]:] = 0

        #一个二维数组，计算每行的方差
        _tmp = new_G_dist[i]   
        _index = torch.where(_tmp!=0,1,0)
        _mean = (torch.sum(_tmp, 1) / torch.sum(_index, 1) + 1e-9).unsqueeze(1)
        
        _std = ((_tmp - _mean) ** 2) * _index
        _std = torch.sqrt(_std.sum(1) / (_index.sum(1) + 1e-9))

        new_G_dist[i] = torch.exp(- _tmp / _std) * (1-torch.eye(max_src_length))
        new_G_dist[i,src_len[i]:] = 0
        new_G_dist[i, :, src_len[i]:] = 0


    new_G_dist[torch.isnan(new_G_dist)] = 0
    
    return new_G_time, new_G_dist

def search_road_index(src_gps):
    start_lng = -8.652
    start_lat = 41.142
    end_lng = -8.578
    end_lat = 41.174

    interval = 64

    lng_interval = abs(end_lng - start_lng) / interval   
    log_interval = abs(end_lat - start_lat) / interval

    batchsize, max_src_length, dim = src_gps.shape
    loc_index = torch.zeros((batchsize, max_src_length, dim))

    for i in range(max_src_length):
        ori_lat, ori_lng = src_gps[:,i,0], src_gps[:,i,1]
        
        latitude_index = int(np.floor(abs(ori_lat-start_lat) / log_interval))
        longitude_index = int(np.floor(abs(ori_lng-start_lng) / lng_interval))
        
        if latitude_index < 0:latitude_index = 0
        if latitude_index > interval-1: latitude_index = interval - 1
        if longitude_index < 0: longitude_index = 0
        if longitude_index > interval - 1: longitude_index = interval - 1

        loc_index[:,i,0] = longitude_index
        loc_index[:,i,1] = latitude_index
    
    loc_index = np.where(src_gps==0,0, loc_index)
    print(src_gps)
    print(loc_index)
    exit()
    return loc_index
