import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
import copy

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