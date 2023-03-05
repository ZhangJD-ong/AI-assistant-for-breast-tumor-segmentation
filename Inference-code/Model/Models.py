# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import copy
import math

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)


class RUnet_encoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(RUnet_encoder, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        act1 = nn.Sigmoid()

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.res_1 = conv_block_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()

        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.res_2 = conv_block_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()

        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.res_3 = conv_block_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()

        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.res_4 = conv_block_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.res_bridge = conv_block_3d(self.num_filters * 8, self.num_filters * 16, activation)


    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)  # -> [1, 4, 128, 128, 128]
        res_1 = self.res_1(x)
        f1 = down_1 + res_1
        pool_1 = self.pool_1(f1)  # -> [1, 4, 64, 64, 64]

        down_2 = self.down_2(pool_1)  # -> [1, 8, 64, 64, 64]
        res_2 = self.res_2(pool_1)
        f2 = down_2 + res_2
        pool_2 = self.pool_2(f2)  # -> [1, 8, 32, 32, 32]

        down_3 = self.down_3(pool_2)  # -> [1, 16, 32, 32, 32]
        res_3 = self.res_3(pool_2)
        f3 = down_3 + res_3
        pool_3 = self.pool_3(f3)  # -> [1, 16, 16, 16, 16]

        down_4 = self.down_4(pool_3)  # -> [1, 32, 16, 16, 16]
        res_4 = self.res_4(pool_3)
        f4 = down_4 + res_4
        pool_4 = self.pool_4(f4)  # -> [1, 32, 8, 8, 8]

        # Bridge
        bridge = self.bridge(pool_4)  # -> [1, 128, 4, 4, 4]
        res_bridge = self.res_bridge(pool_4)+bridge

        return f1,f2,f3,f4,res_bridge


class RUnet_decoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(RUnet_decoder, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        act1 = nn.Sigmoid()

        # Down sampling
        self.ins = conv_block_3d(self.in_dim, self.num_filters * 16, activation)


        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.res_up2 = conv_block_3d(self.num_filters * 24, self.num_filters * 8, activation)

        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.res_up3 = conv_block_3d(self.num_filters * 12, self.num_filters * 4, activation)

        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.res_up4 = conv_block_3d(self.num_filters * 6, self.num_filters * 2, activation)

        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, act1)

    def forward(self, f1,f2,f3,f4,res_bridge):

        res_bridge = self.ins(res_bridge)
        trans_2 = self.trans_2(res_bridge)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, f4], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]
        res_up_2 = self.res_up2(concat_2)

        trans_3 = self.trans_3(up_2 + res_up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, f3], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]
        res_up_3 = self.res_up3(concat_3)

        trans_4 = self.trans_4(up_3 + res_up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, f2], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]
        res_up_4 = self.res_up4(concat_4)

        trans_5 = self.trans_5(up_4 + res_up_4)  # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, f1], dim=1)  # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_5)  # -> [1, 3, 128, 128, 128]
        return out

class Attention(nn.Module):
    def __init__(self, Num_heads,hidden_size,Attention_dropout_rate, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = Num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(Attention_dropout_rate)
        self.proj_dropout = Dropout(Attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, hidden_size,MLP_dim,Dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, MLP_dim)
        self.fc2 = Linear(MLP_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(Dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, hidden_size,MLP_dim,Num_heads,Dropout_rate,Attention_dropout_rate, vis):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size,MLP_dim,Dropout_rate)
        self.attn = Attention(Num_heads,hidden_size,Attention_dropout_rate, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, hidden_size,MLP_dim,Num_heads,Dropout_rate,Attention_dropout_rate, Trans_num_layers, vis=False):
        super(Transformer, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(Trans_num_layers):
            layer = Block(hidden_size,MLP_dim,Num_heads,Dropout_rate,Attention_dropout_rate, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class TranRUnet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters,trans_param):
        super(TranRUnet, self).__init__()
        # self.encoder_pre = RUnet_encoder(in_dim,out_dim,num_filters)
        self.encoder_post = RUnet_encoder(in_dim+1, out_dim, num_filters)
        self.decoder = RUnet_decoder(trans_param['hidden_size']*6,out_dim,num_filters)
        self.spatial_transformer = Transformer(trans_param['hidden_size'],trans_param['MLP_dim'],trans_param['Num_heads'],\
                                       trans_param['Dropout_rate'],trans_param['Attention_dropout_rate'], \
                                       trans_param['Trans_num_layers'], vis=False)
        self.temporal_transformer = Transformer(trans_param['hidden_size'],trans_param['MLP_dim'],trans_param['Num_heads'],\
                                       trans_param['Dropout_rate'],trans_param['Attention_dropout_rate'], \
                                       trans_param['Trans_num_layers'], vis=False)

        self.patch_embeddings = nn.Conv3d(in_channels=256,
                                       out_channels=trans_param['hidden_size'],
                                       kernel_size=1)

        self.position_embeddings = nn.Parameter(torch.zeros(1, 72, trans_param['hidden_size']))
        self.time_embeddings = nn.Parameter(torch.zeros(1, 6, trans_param['hidden_size']))

        self.dropout = Dropout(trans_param['Dropout_rate'])

    def forward(self,x0,x1,x2,x3,x4,x5,time_M):
        f1_0, f2_0, f3_0, f4_0, z0 = self.encoder_post(torch.cat([x0,x0-x0],1))
        f1_1, f2_1, f3_1, f4_1, z1 = self.encoder_post(torch.cat([x1,x1-x0],1))
        f1_2, f2_2, f3_2, f4_2, z2 = self.encoder_post(torch.cat([x2,x2-x0],1))
        f1_3, f2_3, f3_3, f4_3, z3 = self.encoder_post(torch.cat([x3,x3-x0],1))
        f1_4, f2_4, f3_4, f4_4, z4 = self.encoder_post(torch.cat([x4,x4-x0],1))
        f1_5, f2_5, f3_5, f4_5, z5 = self.encoder_post(torch.cat([x5,x5-x0],1))


        f1 =f1_2
        f2 =f2_2
        f3 =f3_2
        f4 =f4_2

        z0 = self.patch_embeddings(z0).unsqueeze(1)
        z0 = z0*(time_M[:, 0].view(z0.shape[0], 1, 1, 1, 1,1))
        z1 = self.patch_embeddings(z1).unsqueeze(1)
        z1 = z1 * (time_M[:, 1].view(z1.shape[0], 1, 1, 1, 1, 1))
        z2 = self.patch_embeddings(z2).unsqueeze(1)
        z2 = z2 * (time_M[:, 2].view(z2.shape[0], 1, 1, 1, 1, 1))
        z3 = self.patch_embeddings(z3).unsqueeze(1)
        z3 = z3 * (time_M[:, 3].view(z3.shape[0], 1, 1, 1, 1, 1))
        z4 = self.patch_embeddings(z4).unsqueeze(1)
        z4 = z4 * (time_M[:, 4].view(z4.shape[0], 1, 1, 1, 1, 1))
        z5 = self.patch_embeddings(z5).unsqueeze(1)
        z5 = z5 * (time_M[:, 5].view(z5.shape[0], 1, 1, 1, 1, 1))

        z = torch.cat([z0,z1,z2,z3,z4,z5],1)
        B,T,hidden,h,w,l = z.size()
        z = z.flatten(3)   #B T hidden hwl

        spatial_z = z.flatten(end_dim = 1)  #BT hidden hwl
        z_spatial = spatial_z.transpose(-1, -2)  #BT hwl hidden
        spatial_embeddings = z_spatial + self.position_embeddings
        spatial_embeddings = self.dropout(spatial_embeddings)
        z_s,_ = self.spatial_transformer(spatial_embeddings)  # BT hwl hidden
        z_s = z_s.permute(0, 2, 1)  # BT hidden hwl
        z_s = z_s.contiguous().view(B*T, hidden, h, w, l)
        z_s = z_s.contiguous().view(B, T, hidden, h, w, l) # B T hidden h w l

        z_temporal = z.permute(0, 3,1, 2).flatten(end_dim = 1)  #Bhwl T hidden
        temporal_embeddings = z_temporal + self.time_embeddings
        temporal_embeddings = self.dropout(temporal_embeddings)
        z_t,_ = self.temporal_transformer(temporal_embeddings)  # Bhwl T hidden
        z_t = z_t.permute(0, 2, 1)  # Bhwl hidden T
        z_t = z_t.contiguous().view(B,h*w*l, hidden, T).permute(0, 3, 2, 1) #B T hidden hwl
        z_t = z_t.contiguous().view(B, T, hidden, h, w, l) # B T hidden h w l
        z_final = (z_t+z_s).transpose(1, 2).contiguous().view(B,-1,h,w,l)

        out = self.decoder(f1,f2,f3,f4,z_final)

        return out




if __name__ == '__main__':
    # net = Transformer(768,600,12,0.1,0.1,12,vis = True)
    # a = torch.zeros([10,196,768])
    # y,w = net(a)
    # print(y.shape)
    device = torch.device('cuda:0')
    trans_param = {'hidden_size': 768, 'MLP_dim': 2048, 'Num_heads': 12, \
    'Dropout_rate': 0.1, 'Attention_dropout_rate':0.0, 'Trans_num_layers':12}

    net = TranRUnet(1,1,16,trans_param).to(device)
    a0 = torch.ones([2,1,96,96,32]).to(device)
    a1 = torch.ones([2, 1, 96, 96, 32]).to(device)
    a2 = torch.ones([2, 1, 96, 96, 32]).to(device)
    a3 = torch.ones([2, 1, 96, 96, 32]).to(device)
    a4 = torch.ones([2, 1, 96, 96, 32]).to(device)
    a5 = torch.ones([2, 1, 96, 96, 32]).to(device)
    b = torch.ones([2,6]).to(device)
    b[0,2:5]=0

    y = net(a0,a1,a2,a3,a4,a5,b)
    print(y.shape)



