import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from copy import deepcopy
import copy
from models.diff_module import diff_CSDI
from models.diff_util import diff_forward_x0_constraint, cal_x0_from_noise_ddpm

class Diff_RNTraj(nn.Module):
    def __init__(self, diff_model, diffusion_hyperparams):
        super(Diff_RNTraj, self).__init__()
        self.diff_model = diff_model
        self.diffusion_param = diffusion_hyperparams
        
    def forward(self, spatial_A_trans, SE, src_eid_seqs, src_rate_seqs):
        """
        spatial_A_trans: UTGraph
        SE: pre-trained road segment representation
        src_eid_seqs: road segment sequence of RNTraj
        src_rate_seqs: moving rate of RNTraj
        """
        batchsize, max_src_len = src_eid_seqs.shape
        
        id_embed = SE[src_eid_seqs]  # B, T, d
        input_data = torch.cat((id_embed, src_rate_seqs.unsqueeze(-1)), -1)  # B, T, d+1
        
        diff_noise, const_loss, x0_loss = diff_forward_x0_constraint(self.diff_model, input_data, self.diffusion_param, SE, spatial_A_trans)
        
        return diff_noise, const_loss, x0_loss

    def generate_data(self, spatial_A_trans, SE, batchsize, length, pre_dim):
        
        """Ggenerate data"""
        
        x0 = cal_x0_from_noise_ddpm(self.diff_model, self.diffusion_param, batchsize, length, pre_dim + 1)  # B, T, 65
        x0_road = x0[:,:,:pre_dim]  # B, T, 64
        B, T, F = x0_road.shape

        x0_road_shape = x0_road.reshape(B*T, F)

        x0_abs = x0_road_shape.norm(dim=1)
        SE_abs = SE.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x0_road_shape, SE) / (torch.einsum('i,j->ij', x0_abs, SE_abs) + 1e-6)
        
        sim_matrix = sim_matrix.reshape(B, T, -1)  # B, T, road num    2000 * 20 = 40000 top1->top10
        sim_matrix = sim_matrix.argmax(-1)
        
        rates = x0[:,:,pre_dim]

        return sim_matrix, rates
