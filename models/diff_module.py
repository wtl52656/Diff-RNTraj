import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
import copy

def attentions_diff(query, key, value, mask=None, dropout=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)
    # print(scores.shape, mask.shape)
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)

class conv_attention(nn.Module):
    def __init__(self, hid_dim, kernel_size, padding):
        super(conv_attention, self).__init__()
        self.conv = nn.Conv1d(hid_dim, hid_dim, kernel_size, padding=padding)
    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param x: (batch, T, hid_dim)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)
        # print(scores.shape, mask.shape)
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
        # scores.shape : B, h, T, F
        B,h,T,F = scores.shape
        # print("-----------")
        # print(scores.shape)
        scores = scores.reshape(-1, T, F).permute(0, 2, 1) # B, F, T
        
        scores = self.conv(scores)
        scores = scores.permute(0, 2, 1).reshape(B,-1, T, F)
        # print(scores.shape)
        # exit()
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # p_attn: (batch, N, h, T1, T2)

        return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)

def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head=8, d_model=512, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        # self.attn = conv_attention(self.d_k, 5, 2)
    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        # N = query.size(1)
        # for i in [query, key, value]:
        #     print(i.shape, i.device)
        # (batch, T, d_model) -linear-> (batch, T, d_model) -view-> (batch, T, h, d_k) -permute(2,3)-> (batch, h, T, d_k)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # print(query.shape, key.shape, value.shape)
        # exit()
        # apply attention on all the projected vectors in batch
        # for i in [query, key, value]:
        #     print(i.device)

        x, self.attn = attentions_diff(query, key, value, mask=mask, dropout=self.dropout)
        # print(x.shape)
        # exit()
        # x:(batch, h, T1, d_k)
        # attn:(batch, h, T1, T2)

        x = x.transpose(1, 2).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, -1, self.h * self.d_k)  # (batch, N, T1, d_model)

        return self.linears[-1](x)


class transformer_layer(nn.Module):
    def __init__(self, hid_dim):
        super(transformer_layer, self).__init__()
        self.mul_attention = MultiHeadAttention(8, hid_dim)
        self.FC = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace=True)
        )
        self.norm_attn = nn.LayerNorm(hid_dim)
        self.norm_fc = nn.LayerNorm(hid_dim)
        self.time_trend_start = nn.Conv1d(hid_dim, hid_dim, 5, padding=2)
        self.time_trend_end = nn.Conv1d(hid_dim, hid_dim, 5, padding=2)
    def forward(self, q, k, v, mask=None):
        """
        :param x: (batch, T, hid_dim)
        """
        # x = self.time_trend_start(x.permute(0, 2, 1)).permute(0, 2, 1)

        attn_x = self.mul_attention(q, k, v, mask)
        x = q + attn_x
        x = self.norm_attn(x)

        fc = self.FC(x)
        x = self.norm_fc(x + fc)

        # x = self.time_trend_end(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

class Multi_attention_Transformer(nn.Module):
    def __init__(self, hid_dim, N):
        super(Multi_attention_Transformer, self).__init__()
        self.layers = nn.ModuleList([transformer_layer(hid_dim) for i in range(N)])
    def forward(self, q, k, v, mask=None):
        """
        :param x: (batch, T, hid_dim)
        """
        for layer in self.layers:
            x = layer(q, k, v, mask)
        return x

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=500, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            x = x + self.pe[:, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x.detach())


def get_torch_trans(heads=8, layers=1, channels=64, key_padding_mask=None):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size, padding=0):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class diff_CSDI(nn.Module):
    def __init__(self, channels, inputdim, num_steps, embedding_dim, pre_dim, rdcl):
        super().__init__()
        self.channels = channels
        self.pre_dim = pre_dim
        self.rdcl = rdcl
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps,
            embedding_dim,
        )

        self.input_projection = Conv1d_with_init(self.pre_dim + 1, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_layer = nn.Linear(self.channels, self.pre_dim + 1)
        
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                Residual_block(
                    res_channels=self.channels, skip_channels=self.channels, 
                    dilation=2 ** (n % 4),
                    diffusion_step_embed_dim_out=embedding_dim,
                )
                for n in range(self.rdcl)
            ]
        )
        self.positionencoder = TemporalPositionalEncoding(self.channels)
        
        self.rnn = nn.GRU(inputdim, inputdim)
        self.bn = nn.BatchNorm1d(inputdim)
        self.out_bn = nn.BatchNorm1d(inputdim)
        self.layernorm = nn.LayerNorm(inputdim)
    def forward(self, x, diffusion_step, mask_mat=None):
        """
        x.shape : B, T, F
        """
        
        B, step_length, c = x.shape
        x = x.permute(0, 2, 1)   # B, F, T
        
        x = self.input_projection(x)
        x = F.relu(x).permute(0, 2, 1)  # B, T, F

        diffusion_emb = self.diffusion_embedding(diffusion_step)       # diffusion_emb shape:  B, 1, F

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        
        x = self.output_layer(x)
        
        return x
        
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out

class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation, 
                 diffusion_step_embed_dim_out):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        # the layer-specific fc for diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # dilated conv layer
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)

        # add mel spectrogram upsampler and conditioner conv1x1 layer
        self.upsample_conv2d = torch.nn.ModuleList()
        for s in [16, 16]:
            conv_trans2d = torch.nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            conv_trans2d = torch.nn.utils.weight_norm(conv_trans2d)
            torch.nn.init.kaiming_normal_(conv_trans2d.weight)
            self.upsample_conv2d.append(conv_trans2d)
        self.mel_conv = Conv(80, 2 * self.res_channels, kernel_size=1)  # 80 is mel bands

        # residual conv1x1 layer, connect to next residual layer
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # skip conv1x1 layer, add to all skip outputs through skip connections
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, x, diffusion_step_embed):
        # x, mel_spec, diffusion_step_embed = input_data
        h = x
        

        # add in diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed)
        h += part_t

        h = h.permute(0, 2, 1)
        B, C, L = h.shape
        assert C == self.res_channels
        
        # dilated conv layer
        h = self.dilated_conv_layer(h)


        # gated-tanh nonlinearity
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

        # residual and skip outputs
        res = self.res_conv(out).permute(0, 2, 1)
        assert x.shape == res.shape
        skip = self.skip_conv(out).permute(0, 2, 1)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability