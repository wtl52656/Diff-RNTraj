import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()

def diff_forward_x0_constraint(net, x, diffusion_hyperparams, SE, spatial_A_trans):
    """
    x: vectorized RNTraj
    """
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha = _dh["T"], _dh["alpha_bar"], _dh["alpha"]
    
    # audio = x
    B, C, L = x.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    diffusion_steps_1 = torch.randint(1, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    
    z = std_normal(x.shape)
    z_x1 = std_normal(x.shape)
    
    xt = torch.sqrt(Alpha_bar[diffusion_steps]) * x + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    #这里得到的是噪声, loss函数之一
    pred_noise = net(xt, diffusion_steps.view(B,1))  # predict \epsilon according to \epsilon_\theta
    
    noise_func = nn.MSELoss()
    xt_noise = noise_func(pred_noise, z)  #将预测的x0与真实的x0计算loss
    
    x0_hat = (xt - torch.sqrt(1-Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_hat = x0_hat[:,:,:L-1]  # B, T, F
    x0_loss = noise_func(x0_hat, x[:,:,:L-1])

    id_sim = torch.einsum('btf,nf->btn',x0_hat,SE)
    id_sim = id_sim.argmax(-1)

    mask = spatial_A_trans[id_sim[:, :-1], id_sim[:, 1:]] == 1e-10
    loss = torch.sum(mask).float()
    
    return xt_noise, loss, x0_loss 

def cal_x0_from_noise_ddpm(net, diffusion_hyperparams, batchsize, length,  feature):
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha = _dh["T"], _dh["alpha_bar"], _dh["alpha"]
    
    diff_input = std_normal((batchsize, length, feature))
    
    with torch.no_grad():
        for t in range(T-1, -1, -1):
            
            predict_noise = net(diff_input, t)  
            
            coeff1 = 1 / (Alpha[t] ** 0.5)
            coeff2 = (1 - Alpha[t]) / ((1 - Alpha_bar[t]) ** 0.5)
            diff_input = coeff1 * (diff_input - coeff2 * predict_noise)
            
            if t>0:
                noise = std_normal(diff_input.shape)
                sigma = ( (1 - Alpha_bar[t-1]) / (1 - Alpha_bar[t]) * (1 - Alpha[t]) ) ** 0.5
                diff_input += sigma * noise
                
    return diff_input
