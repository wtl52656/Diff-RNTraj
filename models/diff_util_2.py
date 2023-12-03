import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def rescale(x):
    """
    Rescale a tensor to 0-1
    """

    return (x - x.min()) / (x.max() - x.min())


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl
    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]
    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed), 
                                      torch.cos(_embed)), 1)
    
    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters
    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t-1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
    
    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)
    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3
    
    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)
    with torch.no_grad():
        for t in range(T-1, -1, -1):
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}
    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams):
    """
    Compute the training loss of epsilon and epsilon_theta
    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    
    audio = X
    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    z = std_normal(audio.shape)
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net((transformed_X, diffusion_steps.view(B,1),))  # predict \epsilon according to \epsilon_\theta
    return loss_fn(epsilon_theta, z)


def test_noise(x, diffusion_hyperparams, src_len):
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha = _dh["T"], _dh["alpha_bar"], _dh["alpha"]
    B, C, L = x.shape
    z = std_normal(x.shape)

    for i in range(B):
        z[i,src_len[i]:,] = 0  
        # z_x1[i,src_len[i]:,] = 0  

    xt = torch.sqrt(Alpha_bar[T-1]) * x + torch.sqrt(1-Alpha_bar[T-1]) * z  # compute x_t from q(x_t|x_0)
    print(torch.sqrt(Alpha_bar[T-1]), torch.sqrt(1-Alpha_bar[T-1]))
    print(xt.mean(), xt.std(), xt.max(), xt.min())
    exit()
def diff_forward_ori(net, x, diffusion_hyperparams):
    """
    x: trajectory_embed， 编码器得到的轨迹embed
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
    
    return xt_noise#, pred_x0_loss, pred_x0

def diff_forward_base(net, x, diffusion_hyperparams, SE, spatial_A_trans):
    """
    x: trajectory_embed， 编码器得到的轨迹embed
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
    
    # cal x0_hat loss
    """
    x0_hat = (xt - torch.sqrt(1-Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_hat = x0_hat[:,:,:L-1]
    # 对比损失
    l1 = x0_hat[:,:C-1,:]
    l2 = x0_hat[:,1:,]
    error_loss = noise_func(l1,l2)
    """

    x0_hat = (xt - torch.sqrt(1-Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_hat = x0_hat[:,:,:L-1]  # B, T, F

    id_sim = torch.einsum('btf,nf->btn',x0_hat,SE)
    id_sim = id_sim.argmax(-1)

    first_id = id_sim[:,:-1].reshape(-1)
    end_id = id_sim[:,1:].reshape(-1)
    # lists = list(zip(first_id, end_id))
    # loss = 1 - spatial_A_trans[lists]
    # print(c)
    # exit()
    # # print(spatial_A_trans.device, first_id.device, end_id.device)
    for b in range(first_id.shape[0]):
            if b == 0:
                loss = (1 - spatial_A_trans[first_id[b],end_id[b]])
            else:
                loss += (1 - spatial_A_trans[first_id[b],end_id[b]])
    return xt_noise, loss#, pred_x0_loss, pred_x0

def diff_forward(net, x, diffusion_hyperparams, SE, spatial_A_trans):
    """
    x: trajectory_embed， 编码器得到的轨迹embed
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
    
    # cal x0_hat loss
    """
    x0_hat = (xt - torch.sqrt(1-Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_hat = x0_hat[:,:,:L-1]
    # 对比损失
    l1 = x0_hat[:,:C-1,:]
    l2 = x0_hat[:,1:,]
    error_loss = noise_func(l1,l2)
    """

    x0_hat = (xt - torch.sqrt(1-Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_hat = x0_hat[:,:,:L-1]  # B, T, F

    id_sim = torch.einsum('btf,nf->btn',x0_hat,SE)
    id_sim = id_sim.argmax(-1)

    first_id = id_sim[:,:-1].reshape(-1)
    # mid_id = id_sim[:,1:-1].reshape(-1)
    end_id = id_sim[:,1:].reshape(-1)
    # lists = list(zip(first_id, end_id))
    # loss = 1 - spatial_A_trans[lists]
    # print(c)
    # exit()
    # # print(spatial_A_trans.device, first_id.device, end_id.device)
    for b in range(first_id.shape[0]):
            if b == 0:
                loss = (1 - spatial_A_trans[first_id[b],end_id[b]])
            else:
                loss += (1 - spatial_A_trans[first_id[b],end_id[b]])
    return xt_noise, loss#, pred_x0_loss, pred_x0


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

    for i in range(id_sim.shape[0]):
        for j in range(id_sim.shape[1] - 1):
            if i + j == 0:
                loss = (1 - spatial_A_trans[id_sim[i,j],id_sim[i, j+1]])
            else:
                loss += (1 - spatial_A_trans[id_sim[i,j],id_sim[i, j+1]])
    return xt_noise, loss, x0_loss 


def diff_forward_contrastive(net, x, diffusion_hyperparams, SE, spatial_A_trans):
    """
    x: trajectory_embed， 编码器得到的轨迹embed
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
    
    # cal x0_hat loss
    """
    x0_hat = (xt - torch.sqrt(1-Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_hat = x0_hat[:,:,:L-1]
    # 对比损失
    l1 = x0_hat[:,:C-1,:]
    l2 = x0_hat[:,1:,]
    error_loss = noise_func(l1,l2)
    """

    x0_hat = (xt - torch.sqrt(1-Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_ori = x0_hat[:,:C-1,:L-1].reshape(B*(C-1), -1)  # B, F
    x0_aug = x0_hat[:,1:,:L-1].reshape(B*(C-1), -1) # B, F

    x_abs = x0_ori.norm(dim=1)
    x_aug_abs = x0_aug.norm(dim=1)
    
    T = 0.3
    sim_matrix = torch.einsum('ik,jk->ij', x0_ori, x0_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    batch_size = B*(C-1)
    
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()


    return xt_noise, loss#, pred_x0_loss, pred_x0


def diff_forward_ngram(net, x, target, diffusion_hyperparams, SE, spatial_A_trans):
    """
    x: trajectory_embed， 编码器得到的轨迹embed
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
    
    # cal x0_hat loss
    """
    x0_hat = (xt - torch.sqrt(1-Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_hat = x0_hat[:,:,:L-1]
    # 对比损失
    l1 = x0_hat[:,:C-1,:]
    l2 = x0_hat[:,1:,]
    error_loss = noise_func(l1,l2)
    """

    x0_hat = (xt - torch.sqrt(1-Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_hat = x0_hat[:,:,:L-1]  # B, T, F

    id_sim = torch.einsum('btf,nf->btn',x0_hat,SE)   # B, T, id
    id_sim = id_sim.argmax(-1)

    
    for i in range(id_sim.shape[0]):
        gram_gener = Counter()
        truth = Counter()
        for j in range(id_sim.shape[1] - 3):
            gram_gener[(id_sim[i,j], id_sim[i,j+1], id_sim[i,j+2], id_sim[i,j+3])] += 1
            truth[(target[i,j], target[i,j+1], target[i,j+2], target[i,j+3])] += 1
        print(gram_gener)
        print(truth)
        exit()

    print(id_sim)

    print(id_sim.shape)
    exit()

    id_prob = torch.softmax(id_sim, -1)  #计算每个节点对应的概率 
    id_sim = id_sim.argmax(-1)

    print(id_sim)
    print(id_sim.shape)
    exit()
    return xt_noise, loss#, pred_x0_loss, pred_x0



def T_x0_loss(net, diffusion_hyperparams, batchsize, length,  feature, SE, spatial_A_trans):
    x0_hat = cal_x0_from_noise_ddim(net, diffusion_hyperparams, batchsize, length,  feature)

    x0_hat = x0_hat[:,:,:feature-1]  # B, T, F

    id_sim = torch.einsum('btf,nf->btn',x0_hat,SE)
    id_sim = id_sim.argmax(-1)

    first_id = id_sim[:,:-1].reshape(-1)
    end_id = id_sim[:,1:].reshape(-1)

    # lists = list(zip(first_id, end_id))
    loss = 1 - spatial_A_trans[first_id, end_id]
    for b in range(first_id.shape[0]):
        if b == 0:
            loss = (1 - spatial_A_trans[first_id[b],end_id[b]])
        else:
            loss += (1 - spatial_A_trans[first_id[b],end_id[b]])


    return loss
def print_mean_std(x):
    print(x.mean(), x.std(), x.max(), x.min())


def cal_x0_from_xt(net, x, diffusion_hyperparams, batchsize, length,  feature):
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha, sample_interval = _dh["T"], _dh["alpha_bar"], _dh["alpha"], _dh['sample_interval']
    
    diff_input = std_normal((batchsize, length, feature)) #torch.sqrt(Alpha_bar[T-1]) * x + torch.sqrt(1-Alpha_bar[T-1]) * z  # compute x_t from q(x_t|x_0)
    
    curr_t = 500
    diffusion_steps = torch.randint(curr_t, size=(batchsize,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    diffusion_steps_1 = torch.randint(1, size=(batchsize,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    
    z = std_normal(x.shape)
    z_x1 = std_normal(x.shape)
    
    diff_input = torch.sqrt(Alpha_bar[diffusion_steps]) * x + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    
    with torch.no_grad():
        # n = x.size(0)
        for t in range(curr_t-1, -1, -1):
            # diff_input = diff_input
            # print(diff_input.mean(), diff_input.std(), diff_input,max(), diff_input.min())
            # exit()
            predict_noise = net(diff_input, t)  
            # for i in [predict_noise]:
            #     print(i.mean(), i.std(), i.max(), i.min())

            coeff1 = 1 / (Alpha[t] ** 0.5)
            coeff2 = (1 - Alpha[t]) / ((1 - Alpha_bar[t]) ** 0.5)
            diff_input = coeff1 * (diff_input - coeff2 * predict_noise)
            # diff_input = diff_input
            # print(diff_input.mean(), diff_input.std(), diff_input.max(), diff_input.min())
            # print(predict_noise.mean(), predict_noise.std(), predict_noise.max(), predict_noise.min())
            # print("--------------")
            # if t>0:
            #     noise = std_normal(diff_input.shape)
            #     sigma = ( (1 - Alpha_bar[t-1]) / (1 - Alpha_bar[t]) * (1 - Alpha[t]) ) ** 0.5
            #     diff_input += sigma * noise
            # xt_1 = torch.sqrt(Alpha_bar[t-1]) * predict_x0 + torch.sqrt(1-Alpha_bar[t-1]) * z_t
            # diff_input = xt_1
    return diff_input


def cal_x0_from_noise_ddpm(net, diffusion_hyperparams, batchsize, length,  feature):
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha = _dh["T"], _dh["alpha_bar"], _dh["alpha"]
    
    diff_input = std_normal((batchsize, length, feature)) #torch.sqrt(Alpha_bar[T-1]) * x + torch.sqrt(1-Alpha_bar[T-1]) * z  # compute x_t from q(x_t|x_0)
    
    with torch.no_grad():
        # n = x.size(0)
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



def cal_x0_from_noise_ddim(net, diffusion_hyperparams, batchsize, length,  feature):
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha, sample_interval = _dh["T"], _dh["alpha_bar"], _dh["alpha"], _dh['sample_interval']
    
    diff_input = std_normal((batchsize, length, feature)) #torch.sqrt(Alpha_bar[T-1]) * x + torch.sqrt(1-Alpha_bar[T-1]) * z  # compute x_t from q(x_t|x_0)
    
    a = [0,1,2,5, 10]
    b = list(range(20, T, sample_interval))
    seq = a + b
    if seq[-1]!=T-1:
        seq = seq + [T-1]
    seq_next = [-1] + seq[:-1]

    with torch.no_grad():
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = i 
            next_t = j

            predict_noise = net(diff_input, t)

            c2 = ((1 - Alpha_bar[t] / Alpha_bar[next_t]) * (1 - Alpha_bar[next_t]) / (1 - Alpha[t])).sqrt()

            c1 = torch.sqrt(1 - Alpha_bar[next_t] - c2 ** 2)

            z_c2 = std_normal(predict_noise.shape) #c2对应的误差
            
            x0_t = (diff_input - predict_noise * (1 - Alpha_bar[t]).sqrt()) / Alpha_bar[t].sqrt()
            x_next_t = torch.sqrt(Alpha_bar[next_t]) * x0_t + c1 * predict_noise + c2 * z_c2

            diff_input = x_next_t
    return diff_input



def cal_x0_from_noise_ddim_1(net, diffusion_hyperparams, batchsize, length, feature):
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha, sample_interval = _dh["T"], _dh["alpha_bar"], _dh["alpha"], _dh['sample_interval']
    
    # audio = x

    # B, C, L = x.shape  # B is batchsize, C=1, L is audio length
    # diffusion_steps = torch.zeros(B,1,1).cuda() + (T - 1)  # randomly sample diffusion steps from 1~T
    # z = 
    # length = 101
    diff_input = std_normal((batchsize, length, feature)) #torch.sqrt(Alpha_bar[T-1]) * x + torch.sqrt(1-Alpha_bar[T-1]) * z  # compute x_t from q(x_t|x_0)
    
    a = [0,1,2,5, 10]
    b = list(range(20, T, sample_interval))
    seq = a + b
    if seq[-1]!=T-1:
        seq = seq + [T-1]
    seq_next = [-1] + seq[:-1]

    with torch.no_grad():
        # n = x.size(0)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = i 
            next_t = j

            predict_noise = net(diff_input, t)
            # if i == 0:break
            # #接下来计算DDIM公式中的误差，因为这里模型的结果直接输出的是x0,所以需要先计算xt-1,再根据xt,xt-1计算误差
            # z_t_1 = std_normal(predict_x0.shape)
            # x_t_1 = torch.sqrt(Alpha_bar[t-1]) * predict_x0 + torch.sqrt(1-Alpha_bar[t-1]) * z_t_1  #预测的是x0,采样的到xt-1
            # error_t = (diff_input - torch.sqrt(Alpha[t]) * x_t_1) / torch.sqrt(1 - Alpha[t]) #得到噪声

            # c2 = torch.sqrt((1 - Alpha_bar[next_t]) / (1 - Alpha_bar[t]) * (1 - Alpha_bar[t] / Alpha_bar[next_t]))
            
            c2 = ((1 - Alpha_bar[t] / Alpha_bar[next_t]) * (1 - Alpha_bar[next_t]) / (1 - Alpha[t])).sqrt()

            c1 = torch.sqrt(1 - Alpha_bar[next_t] - c2 ** 2)

            z_c2 = std_normal(predict_noise.shape) #c2对应的误差
            
            x0_t = (diff_input - predict_noise * (1 - Alpha_bar[t]).sqrt()) / Alpha_bar[t].sqrt()
            x_next_t = torch.sqrt(Alpha_bar[next_t]) * x0_t + c1 * predict_noise + c2 * z_c2

            diff_input = x_next_t
    return diff_input


    
def generalized_steps(x, seq, model, b, c, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            t = t.long()
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t, c)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds