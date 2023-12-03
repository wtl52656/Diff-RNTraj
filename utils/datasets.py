import random
from tqdm import tqdm
import os
from chinese_calendar import is_holiday

import numpy as np
import torch
import matplotlib.pyplot as plt

from common.spatial_func import distance
from common.trajectory import get_tid, Trajectory
from .parse_traj import ParseMMTraj
from .save_traj import SaveTraj2MM
from .utils import create_dir
from models.model_utils import load_rid_freqs
import pickle

class Dataset(torch.utils.data.Dataset):
    """
    customize a dataset for PyTorch
    """

    def __init__(self, trajs_dir, user_idx, mbr, norm_grid_poi_dict, norm_grid_rnfea_dict, weather_dict, parameters, debug=True, datatype='train'):
        self.mbr = mbr  # MBR of all trajectories
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span  # time interval between two consecutive points.
        self.online_features_flag = parameters.online_features_flag
        self.src_grid_seqs, self.src_gps_seqs, self.src_pro_feas = [], [], []
        self.src_eid_seqs, self.src_rate_seqs = [], []
        self.src_road_index_seqs, self.src_times_seq = [], []
        self.trg_gps_seqs, self.trg_rids, self.trg_rates = [], [], []
        self.new_tids = []
        self.uid_seqs = []
        # mm_t_ls, mm_index_ls, mm_grid_ls, mm_gps_ls,
        self.trg_t_seqs, self.trg_index_seqs, self.trg_grid_seqs, self.trg_gps_in_seqs = [], [], [], []
        # above should be [num_seq, len_seq(unpadded)]
        # self.get_data(trajs_dir, user_idx, norm_grid_poi_dict, norm_grid_rnfea_dict, weather_dict, parameters.win_size, 
        #               parameters.ds_type, parameters.keep_ratio, debug,  datatype)
        self.load_data(trajs_dir)
        # print(max(self.src_eid_seqs))
        # exit()
    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.src_grid_seqs)

    def __getitem__(self, index):
        """Generate one sample of data"""
        src_grid_seq = self.src_grid_seqs[index]
        src_gps_seq = self.src_gps_seqs[index]
        src_eid_seq = self.src_eid_seqs[index]
        src_rate_seq = self.src_rate_seqs[index]
        src_time_seq = self.src_times_seq[index]
        # src_road_index_seq = self.src_road_index_seqs[index]
        
        # trg_in_t_seq = self.trg_t_seqs[index]
        # trg_in_index_seq = self.trg_index_seqs[index]
        # trg_in_grid_seq = self.trg_grid_seqs[index]
        # trg_in_gps_seq = self.trg_gps_in_seqs[index]

        trg_gps_seq = self.trg_gps_seqs[index]
        trg_rid = self.trg_rids[index]
        trg_rate = self.trg_rates[index]

        src_grid_seq = self.add_token(src_grid_seq)
        src_gps_seq = self.add_token(src_gps_seq)
        src_eid_seq = self.add_token(src_eid_seq)
        src_rate_seq = self.add_token(src_rate_seq)
        src_time_seq = self.add_token(src_time_seq)
        # src_road_index_seq = self.add_token(src_road_index_seq)
        trg_gps_seq = self.add_token(trg_gps_seq)
        trg_rid = self.add_token(trg_rid)
        trg_rate = self.add_token(trg_rate)
        # src_pro_fea = torch.tensor(self.src_pro_feas[index])

        # trg_in_t_seq = self.add_token(trg_in_t_seq)
        # trg_in_index_seq = self.add_token(trg_in_index_seq)
        # trg_in_grid_seq = self.add_token(trg_in_grid_seq)
        # trg_in_gps_seq = self.add_token(trg_in_gps_seq)

        # use_id_seq = self.uid_seqs[index]
        return src_grid_seq, src_gps_seq, src_eid_seq, src_rate_seq, src_time_seq, trg_gps_seq, trg_rid, trg_rate

    def add_token(self, sequence):
        """
        Append start element(sos in NLP) for each sequence. And convert each list to tensor.
        """
        new_sequence = []
        dimension = len(sequence[0])
        start = [0] * dimension  # pad 0 as start of rate sequence
        new_sequence.append(start)
        new_sequence.extend(sequence)
        new_sequence = torch.tensor(new_sequence)
        return new_sequence

    def get_data(self, trajs_dir, user_idx, norm_grid_poi_dict, norm_grid_rnfea_dict,
                 weather_dict, win_size, ds_type, keep_ratio, debug, datatype):
        parser = ParseMMTraj()
        trg_paths = os.listdir(trajs_dir)
        # debug = True
        if debug:
            print(1)
            trg_paths = os.listdir(trajs_dir)[:100]
            num = -1
        else:
            print(2)
            trg_paths = os.listdir(trajs_dir)
            num = -1
        traj_length_lst = [0 for i in range(101)]
        all_sum = 0
        range_sum = 0
        longer_traj = []
        for file_name in tqdm(trg_paths):
            trajs, user_id = parser.parse(os.path.join(trajs_dir, file_name), user_idx)
            user_id_cnt = 0
            for traj in trajs[:num]:
                uid = user_id[user_id_cnt]
                user_id_cnt += 1
                all_sum += 1

                
                # 轨迹的长度设定在20-100范围内, 如果轨迹的长度超过100, 就按照20-100长度内轨迹的比例进行采样, 将长的轨迹划分成短的
                
                if len(traj.pt_list) < 20 or len(traj.pt_list) > 100:
                    continue
                
                # new_tid_ls, mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, mm_t_ls, mm_index_ls, mm_grid_ls, mm_gps_ls,\
                # ls_grid_seq_ls, ls_gps_seq_ls, ls_road_index_seq_ls, ls_eid_seq_ls, ls_rate_seq_ls, features_ls = self.parse_traj(traj, user_id, norm_grid_poi_dict, norm_grid_rnfea_dict,
                #                                                              weather_dict, win_size, ds_type, keep_ratio)
                
                new_tid_ls, mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, \
                    ls_grid_seq_ls, ls_gps_seq_ls, ls_times_ls, ls_eid_seq_ls, ls_rate_seq_ls = self.parse_traj(traj)
                
                if new_tid_ls is not None:
                    self.new_tids.extend(new_tid_ls)
                    self.trg_gps_seqs.extend(mm_gps_seq_ls)
                    self.trg_rids.extend(mm_eids_ls)
                    self.trg_rates.extend(mm_rates_ls)
                    self.src_grid_seqs.extend(ls_grid_seq_ls)
                    self.src_gps_seqs.extend(ls_gps_seq_ls)
                    
                    # self.src_road_index_seqs.extend(ls_road_index_seq_ls)
                    # self.src_pro_feas.extend(features_ls)
                    self.src_eid_seqs.extend(ls_eid_seq_ls)
                    self.src_rate_seqs.extend(ls_rate_seq_ls)
                    self.src_times_seq.extend(ls_times_ls)
                    # self.trg_t_seqs.extend(mm_t_ls)
                    # self.trg_index_seqs.extend(mm_index_ls)
                    # self.trg_grid_seqs.extend(mm_grid_ls)
                    # self.trg_gps_in_seqs.extend(mm_gps_ls)
                    for traj_length in range(len(new_tid_ls)):
                        self.uid_seqs.append(uid)  # 有的时候一条轨迹会拆分为多条，需要加上相应的uid
                    #  self.trg_index, self.trg_grid, self.trg_gps
                    assert len(new_tid_ls) == len(mm_gps_seq_ls) == len(mm_eids_ls) == len(mm_rates_ls)
        assert len(self.new_tids) == len(self.trg_gps_seqs) == len(self.trg_rids) == len(self.trg_rates) == \
               len(self.src_gps_seqs) == len(self.src_grid_seqs) == len(self.src_eid_seqs) == len(self.src_rate_seqs), \
        'The number of source and target sequence must be equal.'
        save_path = '/data2/WeiTongLong/trajectory/trajetory_generate/dataset/Porto/' + datatype
        with open(save_path + '/eid_seqs.bin', 'wb') as f:  #路段序列
            pickle.dump(self.src_eid_seqs, f)
            f.close()
        with open(save_path + '/rate_seqs.bin', 'wb') as f: #移动率序列
            pickle.dump(self.src_rate_seqs, f)
            f.close()
        with open(save_path + '/times_seq.bin', 'wb') as f:  #轨迹序列
            pickle.dump(self.src_times_seq, f)
            f.close()
        with open(save_path + '/grid_seqs.bin', 'wb') as f:  #GPS对应的网格序列
            pickle.dump(self.src_grid_seqs, f)
            f.close()
        with open(save_path + '/GPS_seqs.bin', 'wb') as f:  #GPS点序列(路网匹配后的)
            pickle.dump(self.trg_gps_seqs, f)
            f.close()
        # exit()
    def load_data(self, trajs_dir):

        with open(trajs_dir + 'eid_seqs.bin', 'rb') as f:  #路段序列
            self.trg_rids = pickle.load(f)
            f.close()
        with open(trajs_dir + 'eid_seqs.bin', 'rb') as f:  #路段序列
            self.src_eid_seqs = pickle.load(f)
            f.close()
        with open(trajs_dir + 'rate_seqs.bin', 'rb') as f: #移动率序列
            self.trg_rates = pickle.load(f)
            f.close()
        with open(trajs_dir + 'rate_seqs.bin', 'rb') as f: #移动率序列
            self.src_rate_seqs = pickle.load(f)
            f.close()
        with open(trajs_dir + 'times_seq.bin', 'rb') as f:  #轨迹序列
            self.src_times_seq = pickle.load(f)
            f.close()
        with open(trajs_dir + 'grid_seqs.bin', 'rb') as f:  #GPS对应的网格序列
            self.src_grid_seqs = pickle.load(f)
            f.close()
        with open(trajs_dir + 'GPS_seqs.bin', 'rb') as f:  #GPS点序列(路网匹配后的)
            self.trg_gps_seqs = pickle.load(f)
            f.close()
        with open(trajs_dir + 'GPS_seqs.bin', 'rb') as f:  #GPS点序列(路网匹配后的)
            self.src_gps_seqs = pickle.load(f)
            f.close()
    def parse_traj(self, traj):
        """
        Split traj based on length.
        Preprocess ground truth (map-matched) Trajectory(), get gps sequence, rid list and rate list.
        Down sample original Trajectory(), get ls_gps, ls_grid sequence and profile features
        Args:
        -----
        traj:
            Trajectory()
        win_size:
            window size of length for a single high sampling trajectory
        ds_type:
            ['uniform', 'random']
            uniform: sample GPS point every down_steps element.
                     the down_step is calculated by 1/remove_ratio
            random: randomly sample (1-down_ratio)*len(old_traj) points by ascending.
        keep_ratio:
            float. range in (0,1). The ratio that keep GPS points to total points.
        Returns:
        --------
        new_tid_ls, mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, ls_grid_seq_ls, ls_gps_seq_ls, features_ls
        """
        # new_trajs = self.get_win_trajs(traj, win_size)
        new_tid_ls = []
        mm_gps_seq_ls, mm_eids_ls, mm_rates_ls = [], [], []
        ls_grid_seq_ls, ls_gps_seq_ls, features_ls = [], [], []
        ls_eid_seq_ls, ls_rate_seq_ls = [], []
        ls_times_ls = []
        mm_t_ls, mm_index_ls, mm_grid_ls, mm_gps_ls = [], [], [], []
        
        # for tr in traj:
            # print(tr)
            # tmp_pt_list = tr.pt_list
        # print(traj)
        new_tid_ls.append(traj.tid)
        # get target sequence
        mm_gps_seq, mm_eids, mm_rates, \
            src_gps_seq, src_grid_seq, src_time_seq, src_eids, src_rates = self.get_trg_seq(traj.pt_list)
        if mm_eids is None:
            return None, None, None, None, None, None, None, None, None

        # # get source sequence
        # ds_pt_list = tr.pt_list #self.downsample_traj(tmp_pt_list, ds_type, keep_ratio)
        # ls_grid_seq, ls_gps_seq, ls_eid_seq, ls_rate_seq, hours, ttl_t, ls_road_index_seq = self.get_src_seq(ds_pt_list, norm_grid_poi_dict, norm_grid_rnfea_dict)
        # features = self.get_pro_features(ds_pt_list, hours, weather_dict)


        # # get target sequence time and grid
        # trg_t, trg_index, trg_grid, trg_gps = self.get_trg_grid_t(tmp_pt_list, ds_pt_list)
        # # check if src and trg len equal, if not return none
        # if len(mm_gps_seq) != ttl_t:
        #     return None, None, None, None, None, None, None, None, None, None, None, None, None, None
        
        mm_gps_seq_ls.append(mm_gps_seq)
        mm_eids_ls.append(mm_eids)
        mm_rates_ls.append(mm_rates)
        ls_grid_seq_ls.append(src_grid_seq)
        ls_gps_seq_ls.append(src_gps_seq)
        # features_ls.append(features)
        ls_eid_seq_ls.append(src_eids)
        ls_times_ls.append(src_time_seq)
        ls_rate_seq_ls.append(src_rates)
            # mm_t_ls.append(trg_t)
            # mm_index_ls.append(trg_index)
            # mm_grid_ls.append(trg_grid)
            # mm_gps_ls.append(trg_gps)
        return new_tid_ls, mm_gps_seq_ls, mm_eids_ls, mm_rates_ls, ls_grid_seq_ls, ls_gps_seq_ls, ls_times_ls, ls_eid_seq_ls, ls_rate_seq_ls

    
    def get_win_trajs(self, traj, win_size):
        pt_list = traj.pt_list
        len_pt_list = len(pt_list)
        if len_pt_list < win_size:
            return [traj]

        num_win = len_pt_list // win_size
        last_traj_len = len_pt_list % win_size + 1
        new_trajs = []
        for w in range(num_win):
            # if last window is large enough then split to a single trajectory
            if w == num_win and last_traj_len > 15:
                tmp_pt_list = pt_list[win_size * w - 1:]
            # elif last window is not large enough then merge to the last trajectory
            elif w == num_win - 1 and last_traj_len <= 15:
                # fix bug, when num_win = 1
                ind = 0
                if win_size * w - 1 > 0:
                    ind = win_size * w - 1
                tmp_pt_list = pt_list[ind:]
            # else split trajectories based on the window size
            else:
                tmp_pt_list = pt_list[max(0, (win_size * w - 1)):win_size * (w + 1)]
                # -1 to make sure the overlap between two trajs

            new_traj = Trajectory(traj.oid, get_tid(traj.oid, tmp_pt_list), tmp_pt_list)
            new_trajs.append(new_traj)
        return new_trajs

    def get_trg_seq(self, pt_list):
        mm_gps_seq = []
        mm_eids = []
        mm_rates = []

        src_grid_seq = []
        src_gps_seq = []
        src_time_seq = []
        
        first_pt = pt_list[0]
        last_pt = pt_list[-1]
        time_interval = self.time_span
        ttl_t = self.get_noramlized_t(first_pt, last_pt, time_interval)
        
        for pt in pt_list:
            candi_pt = pt.data['candi_pt']
            if candi_pt is None:
                return None, None, None, None, None, None, None, None
            else:
                mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
                mm_eids.append([candi_pt.eid])  # keep the same format as seq
                mm_rates.append([candi_pt.rate])

                src_gps_seq.append([pt.lat, pt.lng])

                # src_lng_index, src_lat_index = self.cal_index_lng_lat(pt.lng, pt.lat)
                _lat_grid, _lng_grid = self.gps2grid(pt, self.mbr, self.grid_size)
                src_grid_seq.append([_lat_grid, _lng_grid])

                t = pt.time.hour * 60 * 60 + pt.time.minute * 60 + pt.time.second
                src_time_seq.append([t])
        src_eids = mm_eids
        src_rates = mm_rates
        return mm_gps_seq, mm_eids, mm_rates, src_gps_seq, src_grid_seq, src_time_seq, src_eids, src_rates

    def get_trg_grid_t(self, all_pt_list, downsample_pt_list):
        trg_t = []
        trg_grid = []
        trg_index = []
        trg_gps = []
        first_pt = all_pt_list[0]
        last_pt = all_pt_list[-1]
        time_interval = self.time_span
        ttl_t = self.get_noramlized_t(first_pt, last_pt, time_interval)
        cnt = -1
        for trg_pt in all_pt_list:
            t = self.get_noramlized_t(first_pt, trg_pt, time_interval)
            trg_t.append([t])
            # if trg_pt in downsample_pt_list:
            result = filter(lambda x: x.time == trg_pt.time, downsample_pt_list)  # 这里是根据时间判断在不在downsample_pt_list，直接使用in在列表中查会有问题，不知道为什么会出现这个问题
            result_list = list(result)
            if len(result_list) > 0:
                # if len(trg_index) !=0 and trg_index[-1] == [1] and trg_pt!= all_pt_list[-1]:
                #     # print(downsample_pt_list)
                #     # print(all_pt_list[cnt], trg_pt)

                #     print(downsample_pt_list)   
                #     for aa in all_pt_list:
                #         if aa in downsample_pt_list:
                #             print(aa)
                #     exit()
                # print(trg_pt)
                trg_index.append([1])
                candi_pt = trg_pt.data['candi_pt']
                _lat_grid, _lng_grid = self.gps2grid(candi_pt, self.mbr, self.grid_size)
                trg_gps.append([candi_pt.lat, candi_pt.lng])
                trg_grid.append([_lat_grid, _lng_grid])
            else:
                trg_index.append([0])    
                trg_grid.append([0,0])
                trg_gps.append([0,0])
            cnt += 1
        # exit()
        assert len(trg_t) == len(trg_grid) == len(trg_index) == len(trg_gps), \
        'The number of get_trg_grid_t must be equal.'
        # print(trg_t)
        
        # print(trg_gps)
        # exit()
        #对于缺失的位置，先用前后平均的方法补上
        for i in range(len(trg_index)):
            if trg_index[i] == [0]:
                pre_i = i - 1
                next_i = i + 1
                while True:
                    if trg_index[pre_i]==[1]:break
                    pre_i -= 1
                while True:
                    if trg_index[next_i]==[1]:break
                    next_i += 1
                all_interval, curr_interval = next_i - pre_i, i - pre_i
                all_lat, all_lng = trg_gps[next_i][0] - trg_gps[pre_i][0], trg_gps[next_i][1] - trg_gps[pre_i][1]
                start_lat, start_lng = trg_gps[pre_i][0], trg_gps[pre_i][1]
                moving_lat, moving_lng = all_lat / all_interval * curr_interval, all_lng / all_interval * curr_interval
                trg_gps[i] = [start_lat + moving_lat, start_lng + moving_lng]
                _x, _y = self.gps2grid(i, self.mbr, self.grid_size, trg_new_grid=True, lat=trg_gps[i][0], lng=trg_gps[i][1])
                trg_grid[i] = [_x,_y]
        
        return trg_t, trg_index, trg_grid, trg_gps

    def cal_index_lng_lat(self, lng, lat):
        start_lng = -8.652
        start_lat = 41.142
        end_lng = -8.578
        end_lat = 41.174

        interval = 64

        lng_interval = abs(end_lng - start_lng) / interval   
        log_interval = abs(end_lat - start_lat) / interval
        if lng>=start_lng and lng < end_lng and lat>=start_lat and lat<end_lat:
            latitude=int(np.floor(abs(lat-start_lat) / log_interval))
            longitude=int(np.floor(abs(lng-start_lng) / lng_interval))
            return longitude, latitude
        else:
            return 0, 0
    def get_src_seq(self, ds_pt_list, norm_grid_poi_dict, norm_grid_rnfea_dict):
        hours = []
        ls_grid_seq = []
        ls_gps_seq = []
        ls_eid_seq = []
        ls_rate_seq = []
        ls_road_index_seq = []
        first_pt = ds_pt_list[0]
        last_pt = ds_pt_list[-1]
        time_interval = self.time_span
        ttl_t = self.get_noramlized_t(first_pt, last_pt, time_interval)



        for ds_pt in ds_pt_list:
            # exit()
            hours.append(ds_pt.time.hour)
            
            t = self.get_noramlized_t(first_pt, ds_pt, time_interval)
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])
            locgrid_xid, locgrid_yid = self.gps2grid(ds_pt, self.mbr, self.grid_size)
            if self.online_features_flag:
                poi_features = norm_grid_poi_dict[(locgrid_xid, locgrid_yid)]
                rn_features = norm_grid_rnfea_dict[(locgrid_xid, locgrid_yid)]
                ls_grid_seq.append([locgrid_xid, locgrid_yid, t]+poi_features+rn_features)
            else:
                ls_grid_seq.append([locgrid_xid, locgrid_yid, t])
            


            src_lng_index, src_lat_index = self.cal_index_lng_lat(ds_pt.lng, ds_pt.lat)
            ls_road_index_seq.append([ds_pt.time.hour, src_lng_index, src_lat_index])

            if ds_pt.data['candi_pt'] is not None:
                
                ls_eid_seq.append([ds_pt.data['candi_pt'].eid])
                ls_rate_seq.append([ds_pt.data['candi_pt'].rate])
            else:
                ls_eid_seq.append([0])
                ls_rate_seq.append([0])
        #对于ls_eid_seq=0的值，使用最近的元素平均替代？
        # print(ls_eid_seq)
        for i in range(len(ls_eid_seq)):
            if ls_eid_seq[i]!=0:continue
            pre_i = i - 1
            next_i = i+1
            while True:
                if pre_i < 0: pre_i = 0
                if next_i > len(ls_eid_seq) - 1: next_i = len(ls_eid_seq) - 1
                if ls_eid_seq[pre_i]!=0:
                    ls_eid_seq[i] = ls_eid_seq[pre_i]
                    ls_rate_seq[i] = ls_rate_seq[pre_i]
                    break
                if ls_eid_seq[next_i]!=0:
                    ls_eid_seq[i] = ls_eid_seq[next_i]
                    ls_rate_seq[i] = ls_rate_seq[next_i]
                    break
                pre_i -= 1
                next_i += 1
                if pre_i == 0 and next_i == len(ls_eid_seq) - 1:break
        # print(ls_eid_seq)
        # exit()
        return ls_grid_seq, ls_gps_seq, ls_eid_seq, ls_rate_seq, hours, ttl_t, ls_road_index_seq

    
    def get_pro_features(self, ds_pt_list, hours, weather_dict):
        holiday = is_holiday(ds_pt_list[0].time)*1
        day = ds_pt_list[0].time.day
        hour = {'hour': np.bincount(hours).max()}  # find most frequent hours as hour of the trajectory
        # weather = {'weather': weather_dict[(day, hour['hour'])]}
        features = self.one_hot(hour) + [holiday] #self.one_hot(weather) + [holiday]
        return features
    
    
    def gps2grid(self, pt, mbr, grid_size, trg_new_grid=False, lat=0, lng = 0):
        """
        mbr:
            MBR class.
        grid size:
            int. in meter
        """
        LAT_PER_METER = 8.993203677616966e-06
        LNG_PER_METER = 1.1700193970443768e-05
        lat_unit = LAT_PER_METER * grid_size
        lng_unit = LNG_PER_METER * grid_size
        
        max_xid = int((mbr.max_lat - mbr.min_lat) / lat_unit) + 1
        max_yid = int((mbr.max_lng - mbr.min_lng) / lng_unit) + 1
        if trg_new_grid == False:
            lat = pt.lat
            lng = pt.lng
        locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
        locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1
        
        if locgrid_x < 0: locgrid_x = 0
        if locgrid_x > max_xid: locgrid_x = max_xid
        if locgrid_y < 0: locgrid_y = 0
        if locgrid_y > max_yid: locgrid_y = max_yid

        return locgrid_x, locgrid_y
    
    
    def get_noramlized_t(self, first_pt, current_pt, time_interval):
        """
        calculate normalized t from first and current pt
        return time index (normalized time)
        """
        t = int(1+((current_pt.time - first_pt.time).seconds/time_interval))
        return t

    @staticmethod
    def get_distance(pt_list):
        dist = 0.0
        pre_pt = pt_list[0]
        for pt in pt_list[1:]:
            tmp_dist = distance(pre_pt, pt)
            dist += tmp_dist
            pre_pt = pt
        return dist


    @staticmethod
    def downsample_traj(pt_list, ds_type, keep_ratio):
        """
        Down sample trajectory
        Args:
        -----
        pt_list:
            list of Point()
        ds_type:
            ['uniform', 'random']
            uniform: sample GPS point every down_stepth element.
                     the down_step is calculated by 1/remove_ratio
            random: randomly sample (1-down_ratio)*len(old_traj) points by ascending.
        keep_ratio:
            float. range in (0,1). The ratio that keep GPS points to total points.
        Returns:
        -------
        traj:
            new Trajectory()
        """
        assert ds_type in ['uniform', 'random'], 'only `uniform` or `random` is supported'

        old_pt_list = pt_list.copy()
        start_pt = old_pt_list[0]
        end_pt = old_pt_list[-1]

        if ds_type == 'uniform':
            if (len(old_pt_list) - 1) % int(1 / keep_ratio) == 0:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)]
            else:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)] + [end_pt]
        elif ds_type == 'random':
            sampled_inds = sorted(
                random.sample(range(1, len(old_pt_list) - 1), int((len(old_pt_list) - 2) * keep_ratio)))
            new_pt_list = [start_pt] + list(np.array(old_pt_list)[sampled_inds]) + [end_pt]

        return new_pt_list


    

    @staticmethod
    def one_hot(data):
        one_hot_dict = {'hour': 24, 'weekday': 7, 'weather':5}
        for k, v in data.items():
            encoded_data = [0] * one_hot_dict[k]
            encoded_data[v - 1] = 1
        return encoded_data


# Use for DataLoader
def collate_fn(data):
    """
    Reference: https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py
    Creates mini-batch tensors from the list of tuples (src_seq, src_pro_fea, trg_seq, trg_rid, trg_rate).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Sequences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
    -----
    data: list of tuple (src_seq, src_pro_fea, trg_seq, trg_rid, trg_rate), from dataset.__getitem__().
        - src_seq: torch tensor of shape (?,2); variable length.
        - src_pro_fea: torch tensor of shape (1,64) # concatenate all profile features
        - trg_seq: torch tensor of shape (??,2); variable length.
        - trg_rid: torch tensor of shape (??); variable length.
        - trg_rate: torch tensor of shape (??); variable length.
    Returns:
    --------
    src_grid_seqs:
        torch tensor of shape (batch_size, padded_length, 3)
    src_gps_seqs:
        torch tensor of shape (batch_size, padded_length, 3).
    src_pro_feas:
        torch tensor of shape (batch_size, feature_dim) unnecessary to pad
    src_lengths:
        list of length (batch_size); valid length for each padded source sequence.
    trg_seqs:
        torch tensor of shape (batch_size, padded_length, 2).
    trg_rids:
        torch tensor of shape (batch_size, padded_length, 1).
    trg_rates:
        torch tensor of shape (batch_size, padded_length, 1).
    trg_lengths:
        list of length (batch_size); valid length for each padded target sequence.
    """

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)  # get dim for each sequence
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by source sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    # src_grid_seqs, src_gps_seqs, src_road_index_seqs, src_eid_seqs, src_rate_seqs, trg_in_t_seqs, trg_in_index_seqs, trg_in_grid_seqs, trg_in_gps_seqs, src_pro_feas, trg_gps_seqs, trg_rids, trg_rates, use_id_seq = zip(*data)  # unzip data
    src_grid_seqs, src_gps_seqs, src_eid_seqs, src_rate_seqs, src_time_seqs, trg_gps_seqs, trg_rids, trg_rates = zip(*data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_grid_seqs, src_lengths = merge(src_grid_seqs)
    src_gps_seqs, src_lengths = merge(src_gps_seqs)
    # src_road_index_seqs, _ = merge(src_road_index_seqs)
    src_eid_seqs, _ = merge(src_eid_seqs)
    src_rate_seqs, _ = merge(src_rate_seqs)
    src_time_seqs, _ = merge(src_time_seqs)
    # trg_in_t_seqs, _ = merge(trg_in_t_seqs)
    # trg_in_index_seqs, _ = merge(trg_in_index_seqs)
    # trg_in_grid_seqs, _ = merge(trg_in_grid_seqs)
    # trg_in_gps_seqs, _ = merge(trg_in_gps_seqs)
    
    # src_pro_feas = torch.tensor([list(src_pro_fea) for src_pro_fea in src_pro_feas])
    trg_gps_seqs, trg_lengths = merge(trg_gps_seqs)
    trg_rids, _ = merge(trg_rids)
    trg_rates, _ = merge(trg_rates)

    return src_grid_seqs, src_gps_seqs, src_eid_seqs, src_rate_seqs, src_time_seqs, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths

class LoadData():
    def __init__(self, file_path):
        self.load_data(file_path)

    def load_alldata(self, file_path):
        with open(trajs_dir + 'eid_seqs.bin', 'rb') as f:  #路段序列
            self.src_eid_seqs = pickle.load(f)
            f.close()
        with open(trajs_dir + 'rate_seqs.bin', 'rb') as f:  #路段序列
            self.src_rate_seqs = pickle.load(f)
            f.close()
    
    # def trajs_lst
    # def next_epoch(self, batchsize):

