import numpy as np
import random

import torch
import torch.nn as nn
from tqdm import tqdm
from models.model_utils import toseq, get_constraint_mask
from models.loss_fn import cal_id_acc, check_rn_dis_loss, cal_id_acc_train
from models.trajectory_graph import build_graph,search_road_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('multi_task device', device)


def init_weights(self):
    """
    Here we reproduce Keras default initialization weights for consistency with Keras version
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)

# def int2bit(x):
#     # x: B,T
#     b = bin(x)
#     print(b)
#     exit()

def next_batch(eids, rates):
    length = len(eids)
    for i in range(length):
        yield eids[i], rates[i]
def train(model, spatial_A_trans, SE, all_eids, all_rates, optimizer, log_vars, parameters, diffusion_hyperparams):
    model.train()  # not necessary to have this line but it's safe to use model.train() to train model

    criterion_reg = nn.MSELoss()
    criterion_ce = nn.NLLLoss()
    criterion_ce1 = nn.NLLLoss()
    epoch_ttl_loss = 0
    epoch_const_loss = 0
    epoch_diff_loss = 0
    epoch_x0_loss = 0

    all_length = []
    shuffle_all_eids, shuffle_all_rates = {}, {}
    all_batch_id, all_batch_rate = [],[]
    for k, eids_group in all_eids.items():
        all_length.append(k)
        traj_num = len(eids_group) // parameters.batch_size
        for i in range(traj_num):
            all_batch_id.append(all_eids[k][i*parameters.batch_size:(i+1)*parameters.batch_size])
            all_batch_rate.append(all_rates[k][i*parameters.batch_size:(i+1)*parameters.batch_size])
        if traj_num * parameters.batch_size != len(eids_group):
            all_batch_id.append(all_eids[k][traj_num*parameters.batch_size:])
            all_batch_rate.append(all_rates[k][traj_num*parameters.batch_size:])
    
    # 打乱数据
    zipped = zip(all_batch_id, all_batch_rate)
    zipped_list = list(zipped)
    random.shuffle(zipped_list)
    all_batch_id, all_batch_rate = zip(*zipped_list)

    SE = SE.to(device)
    
    cnt = 0
    for ids, rates in next_batch(all_batch_id, all_batch_rate):
        cnt += 1
        import time
        
        src_eid_seqs = torch.tensor(ids).long().to(device)
        src_rate_seqs = torch.tensor(rates).to(device)
        src_rate_seqs = 2 * src_rate_seqs - 1
        
        spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
        
        
        curr_time = time.time()
        optimizer.zero_grad()
        diff_loss, const_loss, x0_loss = model(spatial_A_trans, SE, src_eid_seqs, src_rate_seqs)  # T, B, id_size   and    T, B, 1
        

        ttl_loss = diff_loss + const_loss + x0_loss 

        curr_time = time.time()
        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)  # log_vars are not necessary to clip
        optimizer.step()
        
        epoch_ttl_loss += ttl_loss.item()
        epoch_const_loss += const_loss.item()
        epoch_diff_loss += diff_loss.item()
        epoch_x0_loss += x0_loss.item()
        
    return log_vars, epoch_ttl_loss / cnt, epoch_const_loss / cnt, epoch_diff_loss / cnt, epoch_x0_loss / cnt

def evaluate(model, spatial_A_trans, road_condition, SE, iterator, rn_dict, grid_rn_dict, rn, raw2new_rid_dict,
             online_features_dict, rid_features_dict, raw_rn_dict, new2raw_rid_dict, parameters, test_flag=False):
    model.eval()  # must have this line since it will affect dropout and batch normalization
    epoch_dis_mae_loss = 0
    epoch_dis_rmse_loss = 0
    epoch_dis_rn_mae_loss = 0
    epoch_dis_rn_rmse_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_rate_loss = 0
    epoch_id_loss = 0 # loss from dl model
    criterion_ce = nn.NLLLoss()
    criterion_reg = nn.MSELoss()

    with torch.no_grad():  # this line can help speed up evaluation
        for i, batch in enumerate(iterator):
            import time
            curr_time = time.time()
            src_grid_seqs, src_gps_seqs, src_eid_seqs, src_rate_seqs, src_time_seqs, src_lengths, \
                    trg_gps_seqs, trg_rids, trg_rates, trg_lengths = batch


            SE = SE.to(device)
            src_grid_seqs = src_grid_seqs.to(device)
            src_eid_seqs = src_eid_seqs.long().to(device)
            src_rate_seqs = src_rate_seqs.to(device)
            src_time_seqs = src_time_seqs.to(device)
            # src_road_index_seqs = src_road_index_seqs.long().to(device)
            # trg_in_grid_seqs = trg_in_grid_seqs.to(device)
            trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
            trg_rates = trg_rates.permute(1, 0, 2).to(device)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
            
            output_ids, output_rates = model.evaluate(spatial_A_trans, SE, src_lengths, src_grid_seqs, src_eid_seqs, src_rate_seqs, src_time_seqs,
                    trg_rids, trg_rates, trg_lengths, teach_force = 0)
            
            curr_time = time.time()

            output_rates = output_rates.squeeze(2)
            if test_flag:
                output_seqs = toseq(rn_dict, output_ids, output_rates, parameters)
            trg_rids = trg_rids.squeeze(2)
            
            trg_rates = trg_rates.squeeze(2)
            if test_flag == False:
                loss_ids1, recall, precision = cal_id_acc_train(output_ids[1:], trg_rids[1:], trg_lengths, True)
            else:
                loss_ids1, recall, precision = cal_id_acc(output_ids[1:], trg_rids[1:], trg_lengths)
            # distance loss
            dis_mae_loss, dis_rmse_loss, dis_rn_mae_loss, dis_rn_rmse_loss = 0, 0, 0, 0
            if test_flag:
                dis_mae_loss, dis_rmse_loss, dis_rn_mae_loss, dis_rn_rmse_loss = check_rn_dis_loss(output_seqs[1:],
                                                                                            output_ids[1:],
                                                                                            output_rates[1:],
                                                                                            trg_gps_seqs[1:],
                                                                                            trg_rids[1:],
                                                                                            trg_rates[1:],
                                                                                            trg_lengths,
                                                                                            rn, raw_rn_dict,
                                                                                            new2raw_rid_dict)
            # for bbp
            output_ids_dim = output_ids.shape[-1]
            output_ids = output_ids[1:].reshape(-1,
                                                output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
            trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
            loss_ids = criterion_ce(output_ids, trg_rids)
            # rate loss
            loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
            curr_time = time.time()
            # print("..................................................")
            epoch_dis_mae_loss += dis_mae_loss
            epoch_dis_rmse_loss += dis_rmse_loss
            epoch_dis_rn_mae_loss += dis_rn_mae_loss
            epoch_dis_rn_rmse_loss += dis_rn_rmse_loss
            epoch_id1_loss += loss_ids1
            epoch_recall_loss += recall
            epoch_precision_loss += precision
            epoch_rate_loss += loss_rates.item()
            epoch_id_loss += loss_ids.item()

        return epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
               epoch_precision_loss / len(iterator), \
               epoch_dis_mae_loss / len(iterator), epoch_dis_rmse_loss / len(iterator), \
               epoch_dis_rn_mae_loss / len(iterator), epoch_dis_rn_rmse_loss / len(iterator), \
               epoch_rate_loss / len(iterator), epoch_id_loss / len(iterator)

def bit2int(x):
    # x: B, T, 12
    x[x>0]=1
    x[x<0]=0
    # res = 0
    for i in range(12):
        if i == 0:res = x[:,:,i] * (2**(11-i))
        
        else:res += x[:,:,i] * (2**(11-i))
        # print(res)
    return res
def generate_data(model, spatial_A_trans, rn_dict, parameters, SE):
    model.eval()  # must have this line since it will affect dropout and batch normalization
    SE = SE.to(device)
    spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
    if parameters.dataset == "Chengdu":
        length2num = np.load("/data/WeiTongLong/data/traj_gen/A_new_dataset/chengdu/gen_all/length_distri.npy")
        start_length, end_length = 20, 60
    if parameters.dataset == "Porto":
        length2num = np.load("/data/WeiTongLong/data/traj_gen/A_new_dataset/Porto_0928/gen_all/length_distri.npy")
        start_length, end_length = 20, 100
    # batchsize=1
    traj_all_num = int(length2num.sum())
    # batchsize=1
    with torch.no_grad():  # this line can help speed up evaluation
        for i in tqdm(range(start_length, end_length+1)):
            curr_length =i
            curr_batch = int(length2num[i] / length2num.sum() * traj_all_num)

            while True:
                if curr_batch > 2000: 
                    _curr_batch = 2000
                else:
                    _curr_batch = curr_batch
                # print(_curr_batch, curr_length)
                ids, rates = model.generate_data(spatial_A_trans, SE, _curr_batch, curr_length, parameters.pre_trained_dim)
                rates[rates>1]=1
                rates[rates<-1]=-1
                rates = (rates + 1) / 2
                output_seqs = toseq(rn_dict, ids,rates, parameters, save_txt_num = i)
                curr_batch = curr_batch - _curr_batch
                if curr_batch <= 0: break
                exit()
            # print(output_seqs)
        # exit()

def val(model, spatial_A_trans, road_condition, SE, all_eids, all_rates, optimizer, log_vars, rn_dict, grid_rn_dict, rn,
          raw2new_rid_dict, online_features_dict, rid_features_dict, parameters, diffusion_hyperparams):
    model.train()  # not necessary to have this line but it's safe to use model.train() to train model

    criterion_reg = nn.MSELoss()
    criterion_ce = nn.NLLLoss()
    criterion_ce1 = nn.NLLLoss()
    epoch_ttl_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_train_id_loss = 0
    epoch_rate_loss = 0
    epoch_diff_loss = 0
    epoch_x0_loss = 0
    epoch_length_loss = 0

    all_length = []
    shuffle_all_eids, shuffle_all_rates = {}, {}
    all_batch_id, all_batch_rate = [],[]
    for k, eids_group in all_eids.items():
        all_length.append(k)
        traj_num = len(eids_group) // parameters.batch_size
        for i in range(traj_num):
            all_batch_id.append(all_eids[k][i*parameters.batch_size:(i+1)*parameters.batch_size])
            all_batch_rate.append(all_rates[k][i*parameters.batch_size:(i+1)*parameters.batch_size])
        if traj_num * parameters.batch_size != len(eids_group):
            all_batch_id.append(all_eids[k][traj_num*parameters.batch_size:])
            all_batch_rate.append(all_rates[k][traj_num*parameters.batch_size:])
    
    # 打乱数据
    zipped = zip(all_batch_id, all_batch_rate)
    zipped_list = list(zipped)
    random.shuffle(zipped_list)
    all_batch_id, all_batch_rate = zip(*zipped_list)

    SE = SE.to(device)

    next_batch(all_batch_id, all_batch_rate)
    cnt = 0
    for ids, rates in next_batch(all_batch_id, all_batch_rate):
        cnt += 1
        import time
        
        src_eid_seqs = torch.tensor(ids).long().to(device)
        src_rate_seqs = torch.tensor(rates).to(device)
        
        spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
        
        # use_id_seq = torch.tensor(np.array(use_id_seq), dtype=torch.long).to(device)
        # use_id_seq = use_id_seq.unsqueeze(0)
        
        # print(src_grid_seqs.shape, trg_in_index_seqs.shape, constraint_mat.shape)
        # print(constraint_mat[1].max())
        # exit()
        curr_time = time.time()
        optimizer.zero_grad()
        output_ids, output_rates,pred_length = model.evaluate(spatial_A_trans, SE, src_eid_seqs, src_rate_seqs, teach_force=0,z0_hat=False)  # T, B, id_size   and    T, B, 1
        # diff_loss = 0
        # print(output_ids, )
        # test_out_data(output_ids, output_rates,pred_length, trg_rids, trg_rates, trg_lengths)

        orward_time = time.time() - curr_time
        curr_time = time.time()
        output_rates = output_rates.squeeze(2)
        # trg_rids = trg_rids.squeeze(2)
        
        # trg_rates = trg_rates.squeeze(2)
        src_eid_seqs = src_eid_seqs.permute(1, 0)
        
        loss_ids1, recall, precision = cal_id_acc_train(output_ids, src_eid_seqs)
        # print(loss_ids1, recall, precision)
        # exit()
        # for bbp
        # print(loss_ids1)
        # exit()
        output_ids_dim = output_ids.shape[-1]
        output_ids = output_ids.reshape(-1, output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
        trg_rids = src_eid_seqs.reshape(-1)  # [(trg len - 1) * batch size],
        # print(output_ids.shape, trg_rids.shape)

        # trg_lengths = torch.tensor(trg_lengths).long().to(device) - 1
        # # print(torch.max(trg_lengths))
        # loss_train_length =  criterion_ce1(pred_length, trg_lengths)

        loss_train_ids = criterion_ce(output_ids, trg_rids)
        # print(output_rates[1:].shape, trg_rates[1:].shape)
        # print(output_rates.shape, src_rate_seqs.shape)
        # exit()
        loss_rates = criterion_reg(output_rates, src_rate_seqs.permute(1,0)) * parameters.lambda1
        
        epoch_ttl_loss += 0
        epoch_id1_loss += loss_ids1
        epoch_recall_loss += recall
        epoch_precision_loss += precision
        epoch_train_id_loss += loss_train_ids.item()
        epoch_rate_loss += loss_rates.item()
        epoch_diff_loss += 0
        epoch_x0_loss += 0
        epoch_length_loss += 0 #loss_train_length.item()
    # exit()
    print(cnt)
    return log_vars, epoch_ttl_loss / cnt, epoch_id1_loss / cnt, epoch_recall_loss / cnt, \
           epoch_precision_loss / cnt, epoch_rate_loss / cnt, epoch_train_id_loss / cnt, epoch_diff_loss / cnt, epoch_x0_loss / cnt, epoch_length_loss / cnt





def test_out_data(output_ids, output_rates,pred_length, trg_rids, trg_rates, trg_lengths):
    print(output_rates[:,0,0])
    print(trg_rates[:,0,0])
    # print(pred_length.argmax(-1), trg_lengths)
    exit()