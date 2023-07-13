import torch
import torch.nn as nn
import torch.nn.functional as F
from ShuttleNet.ShuttleNet_layers import EncoderLayer, EncoderLayer2, DecoderLayer, DecoderLayer2, GatedFusionLayer, GatedFusionLayer2, GatedFusionLayer3, GatedFusionLayer_model3_ver2, GatedFusionLayer_model3_ver3, GatedFusionLayer_model3, EncoderLayer_Big, DecoderLayer_Big, GatedFusionLayer_Big, EncoderLayer_Big2, DecoderLayer_Big2, GatedFusionLayer_Big2
from ShuttleNet.ShuttleNet_embedding import PositionalEncoding, PlayerEmbedding, ShotEmbedding, BoolEmbedding, AreaEmbedding
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PAD = 0

def get_pad_mask(seq):
    return (seq != PAD).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

def alternatemerge(seq_A, seq_B, merge_len, player):
    # (batch, seq_len, dim)
    seq_len = seq_A.shape[1]
    merged_seq = torch.zeros(seq_A.shape[0], merge_len, seq_A.shape[2])

    if seq_len * 2 == (merge_len - 1):
        # if seq_len is odd and B will shorter, e.g., merge = 5, A = 3, B = 2
        merged_seq[:, 1::2, :] = seq_A[:, :seq_len, :]
        merged_seq[:, 2::2, :] = seq_B[:, :seq_len, :]
    elif (seq_len * 2 - 1) == merge_len:
        # if seq_len is odd and A will longer, e.g., merge = 5, A = 3, B = 2
        merged_seq[:, ::2, :] = seq_A[:, :seq_len, :]
        merged_seq[:, 1::2, :] = seq_B[:, :merge_len-seq_len, :]
    elif seq_len * 2 == merge_len:
        if player == 'A':
            merged_seq[:, ::2, :] = seq_A[:, :seq_len, :]
            merged_seq[:, 1::2, :] = seq_B[:, :seq_len, :]
        elif player == 'B':
            merged_seq[:, 1::2, :] = seq_A[:, :seq_len, :]
            merged_seq[:, 2::2, :] = seq_B[:, :seq_len-1, :]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return merged_seq.cuda(seq_A.device)

class ShotGenDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round']+1)
        # self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer = DecoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, trg_mask=None, return_attns=False):
        
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
       
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        
        # split player only for masking
        mask_A = input_shot[:, ::2]
        mask_B = input_shot[:, 1::2]

        # triangular mask
        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)
        
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        # print(embedded_area.shape, embedded_shot.shape, embedded_player.shape)

        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        # print(embedded_area.shape)
        h_a = embedded_area + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num
        h_s = embedded_shot + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num

        # h_a = self.convolution(h_a.unsqueeze(1)).squeeze(1)
        # h_s = self.convolution(h_s.unsqueeze(1)).squeeze(1)
        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        decode_output_area = self.dropout(self.position_embedding(h_a, mode='decode'))
        decode_output_shot = self.dropout(self.position_embedding(h_s, mode='decode'))

        # global
        decode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='decode'))
        decode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='decode'))
        decode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='decode'))
        decode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='decode'))

        decode_global_A, dec_slf_attn_A, dec_enc_attn_A, disentangled_weight_A = self.global_layer(decode_output_area_A, decode_output_shot_A, encode_global_A, slf_attn_mask=trg_global_A_mask, return_attns=return_attns)
        
        
        if decode_output_area_B.shape[1] != 0:
            decode_global_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer(decode_output_area_B, decode_output_shot_B, encode_global_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)

        decode_local_output, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, encode_local_output, slf_attn_mask=trg_local_mask, return_attns=return_attns)
        decoder_self_attention_list = dec_slf_attn if return_attns else []
        decoder_encoder_self_attention_list = dec_enc_attn if return_attns else []

        if decode_output_area_B.shape[1] != 0:
            decode_output_A = alternatemerge(decode_global_A, decode_global_A, decode_local_output.shape[1], 'A')
            decode_output_B = alternatemerge(decode_global_B, decode_global_B, decode_local_output.shape[1], 'B')
        else:
            decode_output_A = decode_global_A.clone()
            decode_output_B = torch.zeros(decode_local_output.shape, device=decode_local_output.device)


        decode_output = self.gated_fusion(decode_output_A, decode_output_B, decode_local_output)

        # (batch, seq_len, encode_dim)
        if return_attns:
            return decode_output, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        return decode_output
class ShotGenDecoder2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round']+1)
        # self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer2(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, 
                encode_local_output_area, encode_local_output_shot,
                encode_global_area_A, encode_global_shot_A,
                encode_global_area_B, encode_global_shot_B,
                trg_mask=None, return_attns=False):
        
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
       
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        
        # split player only for masking
        mask_A = input_shot[:, ::2]
        mask_B = input_shot[:, 1::2]

        # triangular mask
        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)
        
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        # print(embedded_area.shape, embedded_shot.shape, embedded_player.shape)

        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        # print(embedded_area.shape)
        h_a = embedded_area + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num
        h_s = embedded_shot + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num

        # h_a = self.convolution(h_a.unsqueeze(1)).squeeze(1)
        # h_s = self.convolution(h_s.unsqueeze(1)).squeeze(1)
        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        decode_output_area = self.dropout(self.position_embedding(h_a, mode='decode'))
        decode_output_shot = self.dropout(self.position_embedding(h_s, mode='decode'))

        # global
        decode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='decode'))
        decode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='decode'))
        decode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='decode'))
        decode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='decode'))

        for i in range(0, self.config['n_layers']):
            decode_output_area_A, decode_output_shot_A, dec_slf_attn_A, dec_enc_attn_A, disentangled_weight_A = self.global_layer_A(decode_output_area_A, decode_output_shot_A, encode_global_area_A, encode_global_shot_A, slf_attn_mask=trg_global_A_mask, return_attns=return_attns)
            if decode_output_area_B.shape[1] != 0:
                decode_output_area_B, decode_output_shot_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer_B(decode_output_area_B, decode_output_shot_B, encode_global_area_B, encode_global_shot_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)

            decode_output_area, decode_output_shot, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, encode_local_output_area, encode_local_output_shot, slf_attn_mask=trg_local_mask, return_attns=return_attns)
            
        decoder_self_attention_list = dec_slf_attn if return_attns else []
        decoder_encoder_self_attention_list = dec_enc_attn if return_attns else []

        if decode_output_area_B.shape[1] != 0:
            decode_output_A_area = alternatemerge(decode_output_area_A, decode_output_area_A, decode_output_area.shape[1], 'A')
            decode_output_A_shot = alternatemerge(decode_output_shot_A, decode_output_shot_A, decode_output_shot.shape[1], 'A')
            decode_output_B_area = alternatemerge(decode_output_area_B, decode_output_area_B, decode_output_area.shape[1], 'B')
            decode_output_B_shot = alternatemerge(decode_output_shot_B, decode_output_shot_B, decode_output_shot.shape[1], 'B')
        else:
            decode_output_A_area = decode_output_area_A.clone()
            decode_output_A_shot = decode_output_shot_A.clone()
            decode_output_B_area = torch.zeros(decode_output_area.shape, device=decode_output_area.device)
            decode_output_B_shot = torch.zeros(decode_output_shot.shape, device=decode_output_shot.device)

        decode_output = self.gated_fusion(decode_output_A_area, decode_output_B_area, decode_output_area, decode_output_A_shot, decode_output_B_shot, decode_output_shot)

        # (batch, seq_len, encode_dim)
        if return_attns:
            return decode_output, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        return decode_output
class ShotGenDecoder_model3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round']+1)
        # self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer_model3(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, 
                encode_local_output_area, encode_local_output_shot,
                encode_global_area_A, encode_global_shot_A,
                encode_global_area_B, encode_global_shot_B,
                trg_mask=None, return_attns=False):
        
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
       
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        
        # split player only for masking
        mask_A = input_shot[:, ::2]
        mask_B = input_shot[:, 1::2]

        # triangular mask
        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)
        
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        # print(embedded_area.shape, embedded_shot.shape, embedded_player.shape)

        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        # print(embedded_area.shape)
        h_a = embedded_area + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num
        h_s = embedded_shot + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num

        # h_a = self.convolution(h_a.unsqueeze(1)).squeeze(1)
        # h_s = self.convolution(h_s.unsqueeze(1)).squeeze(1)
        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        decode_output_area = self.dropout(self.position_embedding(h_a, mode='decode'))
        decode_output_shot = self.dropout(self.position_embedding(h_s, mode='decode'))

        # global
        decode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='decode'))
        decode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='decode'))
        decode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='decode'))
        decode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='decode'))

        for i in range(0, self.config['n_layers']):
            decode_output_area_A, decode_output_shot_A, dec_slf_attn_A, dec_enc_attn_A, disentangled_weight_A = self.global_layer_A(decode_output_area_A, decode_output_shot_A, encode_global_area_A, encode_global_shot_A, slf_attn_mask=trg_global_A_mask, return_attns=return_attns)
            if decode_output_area_B.shape[1] != 0:
                decode_output_area_B, decode_output_shot_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer_B(decode_output_area_B, decode_output_shot_B, encode_global_area_B, encode_global_shot_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)

            decode_output_area, decode_output_shot, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, encode_local_output_area, encode_local_output_shot, slf_attn_mask=trg_local_mask, return_attns=return_attns)
            
        decoder_self_attention_list = dec_slf_attn if return_attns else []
        decoder_encoder_self_attention_list = dec_enc_attn if return_attns else []

        if decode_output_area_B.shape[1] != 0:
            decode_output_A_area = alternatemerge(decode_output_area_A, decode_output_area_A, decode_output_area.shape[1], 'A')
            decode_output_A_shot = alternatemerge(decode_output_shot_A, decode_output_shot_A, decode_output_shot.shape[1], 'A')
            decode_output_B_area = alternatemerge(decode_output_area_B, decode_output_area_B, decode_output_area.shape[1], 'B')
            decode_output_B_shot = alternatemerge(decode_output_shot_B, decode_output_shot_B, decode_output_shot.shape[1], 'B')
        else:
            decode_output_A_area = decode_output_area_A.clone()
            decode_output_A_shot = decode_output_shot_A.clone()
            decode_output_B_area = torch.zeros(decode_output_area.shape, device=decode_output_area.device)
            decode_output_B_shot = torch.zeros(decode_output_shot.shape, device=decode_output_shot.device)

        decode_output_area, decode_output_shot = self.gated_fusion(decode_output_A_area, decode_output_B_area, decode_output_area, decode_output_A_shot, decode_output_B_shot, decode_output_shot)

        # (batch, seq_len, encode_dim)
        if return_attns:
            return decode_output_area, decode_output_shot, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        return decode_output_area, decode_output_shot
class ShotGenDecoder_model3_ver2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round']+1)
        # self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer_model3_ver2(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, 
                encode_local_output_area, encode_local_output_shot,
                encode_global_area_A, encode_global_shot_A,
                encode_global_area_B, encode_global_shot_B,
                trg_mask=None, return_attns=False):
        
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
       
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        
        # split player only for masking
        mask_A = input_shot[:, ::2]
        mask_B = input_shot[:, 1::2]

        # triangular mask
        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)
        
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        # print(embedded_area.shape, embedded_shot.shape, embedded_player.shape)

        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        # print(embedded_area.shape)
        h_a = embedded_area + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num
        h_s = embedded_shot + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num

        # h_a = self.convolution(h_a.unsqueeze(1)).squeeze(1)
        # h_s = self.convolution(h_s.unsqueeze(1)).squeeze(1)
        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        decode_output_area = self.dropout(self.position_embedding(h_a, mode='decode'))
        decode_output_shot = self.dropout(self.position_embedding(h_s, mode='decode'))

        # global
        decode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='decode'))
        decode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='decode'))
        decode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='decode'))
        decode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='decode'))

        for i in range(0, self.config['n_layers']):
            decode_output_area_A, decode_output_shot_A, dec_slf_attn_A, dec_enc_attn_A, disentangled_weight_A = self.global_layer(decode_output_area_A, decode_output_shot_A, encode_global_area_A, encode_global_shot_A, slf_attn_mask=trg_global_A_mask, return_attns=return_attns)
            if decode_output_area_B.shape[1] != 0:
                decode_output_area_B, decode_output_shot_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer(decode_output_area_B, decode_output_shot_B, encode_global_area_B, encode_global_shot_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)

            decode_output_area, decode_output_shot, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, encode_local_output_area, encode_local_output_shot, slf_attn_mask=trg_local_mask, return_attns=return_attns)
            
        decoder_self_attention_list = dec_slf_attn if return_attns else []
        decoder_encoder_self_attention_list = dec_enc_attn if return_attns else []

        if decode_output_area_B.shape[1] != 0:
            decode_output_A_area = alternatemerge(decode_output_area_A, decode_output_area_A, decode_output_area.shape[1], 'A')
            decode_output_A_shot = alternatemerge(decode_output_shot_A, decode_output_shot_A, decode_output_shot.shape[1], 'A')
            decode_output_B_area = alternatemerge(decode_output_area_B, decode_output_area_B, decode_output_area.shape[1], 'B')
            decode_output_B_shot = alternatemerge(decode_output_shot_B, decode_output_shot_B, decode_output_shot.shape[1], 'B')
        else:
            decode_output_A_area = decode_output_area_A.clone()
            decode_output_A_shot = decode_output_shot_A.clone()
            decode_output_B_area = torch.zeros(decode_output_area.shape, device=decode_output_area.device)
            decode_output_B_shot = torch.zeros(decode_output_shot.shape, device=decode_output_shot.device)

        decode_output_area, decode_output_shot = self.gated_fusion(decode_output_A_area, decode_output_B_area, decode_output_area, decode_output_A_shot, decode_output_B_shot, decode_output_shot)

        # (batch, seq_len, encode_dim)
        if return_attns:
            return decode_output_area, decode_output_shot, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        return decode_output_area, decode_output_shot
class ShotGenDecoder_model3_ver3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round']+1)
        # self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer_model3_ver3(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, 
                encode_local_output_area, encode_local_output_shot,
                encode_global_area_A, encode_global_shot_A,
                encode_global_area_B, encode_global_shot_B,
                trg_mask=None, return_attns=False):
        
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
       
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        
        # split player only for masking
        mask_A = input_shot[:, ::2]
        mask_B = input_shot[:, 1::2]

        # triangular mask
        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)
        
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        # print(embedded_area.shape, embedded_shot.shape, embedded_player.shape)

        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        # print(embedded_area.shape)
        h_a = embedded_area + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num
        h_s = embedded_shot + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num

        # h_a = self.convolution(h_a.unsqueeze(1)).squeeze(1)
        # h_s = self.convolution(h_s.unsqueeze(1)).squeeze(1)
        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        decode_output_area = self.dropout(self.position_embedding(h_a, mode='decode'))
        decode_output_shot = self.dropout(self.position_embedding(h_s, mode='decode'))

        # global
        decode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='decode'))
        decode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='decode'))
        decode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='decode'))
        decode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='decode'))

        for i in range(0, self.config['n_layers']):
            decode_output_area_A, decode_output_shot_A, dec_slf_attn_A, dec_enc_attn_A, disentangled_weight_A = self.global_layer(decode_output_area_A, decode_output_shot_A, encode_global_area_A, encode_global_shot_A, slf_attn_mask=trg_global_A_mask, return_attns=return_attns)
            if decode_output_area_B.shape[1] != 0:
                decode_output_area_B, decode_output_shot_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer(decode_output_area_B, decode_output_shot_B, encode_global_area_B, encode_global_shot_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)

            decode_output_area, decode_output_shot, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, encode_local_output_area, encode_local_output_shot, slf_attn_mask=trg_local_mask, return_attns=return_attns)
            
        decoder_self_attention_list = dec_slf_attn if return_attns else []
        decoder_encoder_self_attention_list = dec_enc_attn if return_attns else []

        if decode_output_area_B.shape[1] != 0:
            decode_output_A_area = alternatemerge(decode_output_area_A, decode_output_area_A, decode_output_area.shape[1], 'A')
            decode_output_A_shot = alternatemerge(decode_output_shot_A, decode_output_shot_A, decode_output_shot.shape[1], 'A')
            decode_output_B_area = alternatemerge(decode_output_area_B, decode_output_area_B, decode_output_area.shape[1], 'B')
            decode_output_B_shot = alternatemerge(decode_output_shot_B, decode_output_shot_B, decode_output_shot.shape[1], 'B')
        else:
            decode_output_A_area = decode_output_area_A.clone()
            decode_output_A_shot = decode_output_shot_A.clone()
            decode_output_B_area = torch.zeros(decode_output_area.shape, device=decode_output_area.device)
            decode_output_B_shot = torch.zeros(decode_output_shot.shape, device=decode_output_shot.device)

        decode_output = self.gated_fusion(decode_output_A_area, decode_output_B_area, decode_output_area, decode_output_A_shot, decode_output_B_shot, decode_output_shot)

        # (batch, seq_len, encode_dim)
        if return_attns:
            return decode_output, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        return decode_output
class ShotGenDecoder_model3_small(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round']+1)
        # self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer3(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

    def forward(self, input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand,
                encode_local_output_area, encode_local_output_shot,
                encode_global_area_A, encode_global_shot_A,
                encode_global_area_B, encode_global_shot_B,
                trg_mask=None, return_attns=False):
        
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
       
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        
        # split player only for masking
        mask_A = input_shot[:, ::2]
        mask_B = input_shot[:, 1::2]

        # triangular mask
        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)
        
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        # print(embedded_area.shape, embedded_shot.shape, embedded_player.shape)

        # embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        # embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        # embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        # print(embedded_area.shape)
        h_a = embedded_area + embedded_player + embedded_ball_aroundhead + embedded_ball_backhnad
        h_s = embedded_shot + embedded_player + embedded_ball_aroundhead + embedded_ball_backhnad

        # h_a = self.convolution(h_a.unsqueeze(1)).squeeze(1)
        # h_s = self.convolution(h_s.unsqueeze(1)).squeeze(1)
        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        decode_output_area = self.dropout(self.position_embedding(h_a, mode='decode'))
        decode_output_shot = self.dropout(self.position_embedding(h_s, mode='decode'))

        # global
        decode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='decode'))
        decode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='decode'))
        decode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='decode'))
        decode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='decode'))

        for i in range(0, self.config['n_layers']):
            decode_output_area_A, decode_output_shot_A, dec_slf_attn_A, dec_enc_attn_A, disentangled_weight_A = self.global_layer_A(decode_output_area_A, decode_output_shot_A, encode_global_area_A, encode_global_shot_A, slf_attn_mask=trg_global_A_mask, return_attns=return_attns)
            if decode_output_area_B.shape[1] != 0:
                decode_output_area_B, decode_output_shot_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer_B(decode_output_area_B, decode_output_shot_B, encode_global_area_B, encode_global_shot_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)

            decode_output_area, decode_output_shot, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, encode_local_output_area, encode_local_output_shot, slf_attn_mask=trg_local_mask, return_attns=return_attns)
            
        decoder_self_attention_list = dec_slf_attn if return_attns else []
        decoder_encoder_self_attention_list = dec_enc_attn if return_attns else []

        if decode_output_area_B.shape[1] != 0:
            decode_output_A_area = alternatemerge(decode_output_area_A, decode_output_area_A, decode_output_area.shape[1], 'A')
            decode_output_A_shot = alternatemerge(decode_output_shot_A, decode_output_shot_A, decode_output_shot.shape[1], 'A')
            decode_output_B_area = alternatemerge(decode_output_area_B, decode_output_area_B, decode_output_area.shape[1], 'B')
            decode_output_B_shot = alternatemerge(decode_output_shot_B, decode_output_shot_B, decode_output_shot.shape[1], 'B')
        else:
            decode_output_A_area = decode_output_area_A.clone()
            decode_output_A_shot = decode_output_shot_A.clone()
            decode_output_B_area = torch.zeros(decode_output_area.shape, device=decode_output_area.device)
            decode_output_B_shot = torch.zeros(decode_output_shot.shape, device=decode_output_shot.device)

        decode_output_area, decode_output_shot = self.gated_fusion(decode_output_A_area, decode_output_B_area, decode_output_area, decode_output_A_shot, decode_output_B_shot, decode_output_shot)

        # (batch, seq_len, encode_dim)
        if return_attns:
            return decode_output_area, decode_output_shot, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        return decode_output_area, decode_output_shot
class ShotGenDecoderSimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round']+1)
        # self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer3(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

    def forward(self, input_shot, input_x, input_y, input_player, 
                encode_local_output_area, encode_local_output_shot,
                encode_global_area_A, encode_global_shot_A,
                encode_global_area_B, encode_global_shot_B,
                trg_mask=None, return_attns=False):
        
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
       
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        
        # split player only for masking
        mask_A = input_shot[:, ::2]
        mask_B = input_shot[:, 1::2]

        # triangular mask
        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)
        
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        # print(embedded_area.shape, embedded_shot.shape, embedded_player.shape)


        # print(embedded_area.shape)
        h_a = embedded_area + embedded_player
        h_s = embedded_shot + embedded_player
        # h_a = self.convolution(h_a.unsqueeze(1)).squeeze(1)
        # h_s = self.convolution(h_s.unsqueeze(1)).squeeze(1)
        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        decode_output_area = self.dropout(self.position_embedding(h_a, mode='decode'))
        decode_output_shot = self.dropout(self.position_embedding(h_s, mode='decode'))

        # global
        decode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='decode'))
        decode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='decode'))
        decode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='decode'))
        decode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='decode'))

        for i in range(0, self.config['n_layers']):
            decode_output_area_A, decode_output_shot_A, dec_slf_attn_A, dec_enc_attn_A, disentangled_weight_A = self.global_layer_A(decode_output_area_A, decode_output_shot_A, encode_global_area_A, encode_global_shot_A, slf_attn_mask=trg_global_A_mask, return_attns=return_attns)
            if decode_output_area_B.shape[1] != 0:
                decode_output_area_B, decode_output_shot_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer_B(decode_output_area_B, decode_output_shot_B, encode_global_area_B, encode_global_shot_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)

            decode_output_area, decode_output_shot, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, encode_local_output_area, encode_local_output_shot, slf_attn_mask=trg_local_mask, return_attns=return_attns)
            
        decoder_self_attention_list = dec_slf_attn if return_attns else []
        decoder_encoder_self_attention_list = dec_enc_attn if return_attns else []

        if decode_output_area_B.shape[1] != 0:
            decode_output_A_area = alternatemerge(decode_output_area_A, decode_output_area_A, decode_output_area.shape[1], 'A')
            decode_output_A_shot = alternatemerge(decode_output_shot_A, decode_output_shot_A, decode_output_shot.shape[1], 'A')
            decode_output_B_area = alternatemerge(decode_output_area_B, decode_output_area_B, decode_output_area.shape[1], 'B')
            decode_output_B_shot = alternatemerge(decode_output_shot_B, decode_output_shot_B, decode_output_shot.shape[1], 'B')
        else:
            decode_output_A_area = decode_output_area_A.clone()
            decode_output_A_shot = decode_output_shot_A.clone()
            decode_output_B_area = torch.zeros(decode_output_area.shape, device=decode_output_area.device)
            decode_output_B_shot = torch.zeros(decode_output_shot.shape, device=decode_output_shot.device)

        decode_output_area, decode_output_shot = self.gated_fusion(decode_output_A_area, decode_output_B_area, decode_output_area, decode_output_A_shot, decode_output_B_shot, decode_output_shot)

        # (batch, seq_len, encode_dim)
        if return_attns:
            return decode_output_area, decode_output_shot, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        return decode_output_area, decode_output_shot
class ShotGenDecoder_Big(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round']+1)
        # self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = DecoderLayer_Big(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = DecoderLayer_Big(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer_Big(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer_Big(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, 
                encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B,
                trg_mask=None, return_attns=False):
        
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
       
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        
        # split player only for masking
        mask_A = input_shot[:, ::2]
        mask_B = input_shot[:, 1::2]

        # triangular mask
        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)
        
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        # print(embedded_area.shape, embedded_shot.shape, embedded_player.shape)

        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        # print(embedded_area.shape)
        # h_a = embedded_area + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num
        # h_s = embedded_shot + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num

        # h_a = self.convolution(h_a.unsqueeze(1)).squeeze(1)
        # h_s = self.convolution(h_s.unsqueeze(1)).squeeze(1)
        # split player

        h_a = embedded_area + embedded_player
        h_s = embedded_shot + embedded_player
        h_bh = embedded_ball_height + embedded_player
        h_ba = embedded_ball_aroundhead + embedded_player
        h_bb = embedded_ball_backhnad + embedded_player
        h_pan = embedded_player_area_num + embedded_player
        h_oan = embedded_opponent_area_num + embedded_player

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]
        h_bh_A = h_bh[:, ::2]
        h_bh_B = h_bh[:, 1::2]
        h_ba_A = h_ba[:, ::2]
        h_ba_B = h_ba[:, 1::2]
        h_bb_A = h_bb[:, ::2]
        h_bb_B = h_bb[:, 1::2]
        h_pan_A = h_pan[:, ::2]
        h_pan_B = h_pan[:, 1::2]
        h_oan_A = h_oan[:, ::2]
        h_oan_B = h_oan[:, 1::2]

        # local
        decode_output_area = self.dropout(self.position_embedding(h_a, mode='encode'))
        decode_output_shot = self.dropout(self.position_embedding(h_s, mode='encode'))
        decode_output_bh = self.dropout(self.position_embedding(h_bh, mode='encode'))
        decode_output_ba = self.dropout(self.position_embedding(h_ba, mode='encode'))
        decode_output_bb = self.dropout(self.position_embedding(h_bb, mode='encode'))
        decode_output_pan = self.dropout(self.position_embedding(h_pan, mode='encode'))
        decode_output_oan = self.dropout(self.position_embedding(h_oan, mode='encode'))
        # global
        decode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='encode'))
        decode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='encode'))
        decode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='encode'))
        decode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='encode'))
        decode_output_bh_A = self.dropout(self.position_embedding(h_bh_A, mode='encode'))
        decode_output_bh_B = self.dropout(self.position_embedding(h_bh_B, mode='encode'))
        decode_output_ba_A = self.dropout(self.position_embedding(h_ba_A, mode='encode'))
        decode_output_ba_B = self.dropout(self.position_embedding(h_ba_B, mode='encode'))
        decode_output_bb_A = self.dropout(self.position_embedding(h_bb_A, mode='encode'))
        decode_output_bb_B = self.dropout(self.position_embedding(h_bb_B, mode='encode'))
        decode_output_pan_A = self.dropout(self.position_embedding(h_pan_A, mode='encode'))
        decode_output_pan_B = self.dropout(self.position_embedding(h_pan_B, mode='encode'))
        decode_output_oan_A = self.dropout(self.position_embedding(h_oan_A, mode='encode'))
        decode_output_oan_B = self.dropout(self.position_embedding(h_oan_B, mode='encode'))

        for i in range(0, self.config['n_layers']):
            decode_output_area_A, decode_output_shot_A, decode_output_bh_A, decode_output_ba_A, decode_output_bb_A, decode_output_pan_A, decode_output_oan_A, dec_slf_attn_A, dec_enc_attn_A, disentangled_weight_A = self.global_layer_A(decode_output_area_A, decode_output_shot_A, decode_output_bh_A, decode_output_ba_A, decode_output_bb_A, decode_output_pan_A, decode_output_oan_A,
                                                                                                                                    encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, slf_attn_mask=trg_global_A_mask, return_attns=return_attns)
            if decode_output_area_B.shape[1] != 0:
                decode_output_area_B, decode_output_shot_B, decode_output_bh_B, decode_output_ba_B, decode_output_bb_B, decode_output_pan_B, decode_output_oan_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer_B(decode_output_area_B, decode_output_shot_B, decode_output_bh_B, decode_output_ba_B, decode_output_bb_B, decode_output_pan_B, decode_output_oan_B,
                                                                                                                                    encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)
                # decode_output_area_B, decode_output_shot_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer_B(decode_output_area_B, decode_output_shot_B, encode_global_area_B, encode_global_shot_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)

            decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan,
                                                                                                                                    encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, slf_attn_mask=trg_local_mask, return_attns=return_attns)
            # decode_output_area, decode_output_shot, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, encode_local_output_area, encode_local_output_shot, slf_attn_mask=trg_local_mask, return_attns=return_attns)
            
        decoder_self_attention_list = dec_slf_attn if return_attns else []
        decoder_encoder_self_attention_list = dec_enc_attn if return_attns else []

        if decode_output_area_B.shape[1] != 0:
            decode_output_A_area = alternatemerge(decode_output_area_A, decode_output_area_A, decode_output_area.shape[1], 'A')
            decode_output_A_shot = alternatemerge(decode_output_shot_A, decode_output_shot_A, decode_output_shot.shape[1], 'A')
            decode_output_A_bh = alternatemerge(decode_output_bh_A, decode_output_bh_A, decode_output_bh.shape[1], 'A')
            decode_output_A_ba = alternatemerge(decode_output_ba_A, decode_output_ba_A, decode_output_ba.shape[1], 'A')
            decode_output_A_bb = alternatemerge(decode_output_bb_A, decode_output_bb_A, decode_output_bb.shape[1], 'A')
            decode_output_A_pan = alternatemerge(decode_output_pan_A, decode_output_pan_A, decode_output_pan.shape[1], 'A')
            decode_output_A_oan = alternatemerge(decode_output_oan_A, decode_output_oan_A, decode_output_oan.shape[1], 'A')
            decode_output_B_area = alternatemerge(decode_output_area_B, decode_output_area_B, decode_output_area.shape[1], 'B')
            decode_output_B_shot = alternatemerge(decode_output_shot_B, decode_output_shot_B, decode_output_shot.shape[1], 'B')
            decode_output_B_bh = alternatemerge(decode_output_bh_B, decode_output_bh_B, decode_output_bh.shape[1], 'B')
            decode_output_B_ba = alternatemerge(decode_output_ba_B, decode_output_ba_B, decode_output_ba.shape[1], 'B')
            decode_output_B_bb = alternatemerge(decode_output_bb_B, decode_output_bb_B, decode_output_bb.shape[1], 'B')
            decode_output_B_pan = alternatemerge(decode_output_pan_B, decode_output_pan_B, decode_output_pan.shape[1], 'B')
            decode_output_B_oan = alternatemerge(decode_output_oan_B, decode_output_oan_B, decode_output_oan.shape[1], 'B')
        else:
            decode_output_A_area = decode_output_area_A.clone()
            decode_output_A_shot = decode_output_shot_A.clone()
            decode_output_A_bh = decode_output_bh_A.clone()
            decode_output_A_ba = decode_output_ba_A.clone()
            decode_output_A_bb = decode_output_bb_A.clone()
            decode_output_A_pan = decode_output_pan_A.clone()
            decode_output_A_oan = decode_output_oan_A.clone()
            decode_output_B_area = torch.zeros(decode_output_area.shape, device=decode_output_area.device)
            decode_output_B_shot = torch.zeros(decode_output_shot.shape, device=decode_output_shot.device)
            decode_output_B_bh = torch.zeros(decode_output_bh.shape, device=decode_output_bh.device)
            decode_output_B_ba = torch.zeros(decode_output_ba.shape, device=decode_output_ba.device)
            decode_output_B_bb = torch.zeros(decode_output_bb.shape, device=decode_output_bb.device)
            decode_output_B_pan = torch.zeros(decode_output_pan.shape, device=decode_output_pan.device)
            decode_output_B_oan = torch.zeros(decode_output_oan.shape, device=decode_output_oan.device)

        decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan = self.gated_fusion(decode_output_A_area, decode_output_B_area, decode_output_area, decode_output_A_shot, decode_output_B_shot, decode_output_shot,
                                                                   decode_output_A_bh, decode_output_B_bh, decode_output_bh, decode_output_A_ba, decode_output_B_ba, decode_output_ba,
                                                                   decode_output_A_bb, decode_output_B_bb, decode_output_bb, decode_output_A_pan, decode_output_B_pan, decode_output_pan,
                                                                   decode_output_A_oan, decode_output_B_oan, decode_output_oan)

        # (batch, seq_len, encode_dim)
        if return_attns:
            return decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        return decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan
class ShotGenDecoder_Big2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.pl_embedding = nn.Linear(2, config['area_dim'])
        self.ol_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round']+1)
        # self.convolution = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = DecoderLayer_Big2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = DecoderLayer_Big2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer_Big2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer_Big2(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, input_player_location_x, input_player_location_y, input_opponent_location_x, input_opponent_location_y,
                encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol,
                encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A,
                encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B,
                trg_mask=None, return_attns=False):
        
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
       
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        pl = torch.cat((input_player_location_x.unsqueeze(-1), input_player_location_y.unsqueeze(-1)), dim=-1).float()
        ol = torch.cat((input_opponent_location_x.unsqueeze(-1), input_opponent_location_y.unsqueeze(-1)), dim=-1).float()
        
        # split player only for masking
        mask_A = input_shot[:, ::2]
        mask_B = input_shot[:, 1::2]

        # triangular mask
        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A)
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B)
        
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        # print(embedded_area.shape, embedded_shot.shape, embedded_player.shape)
        embedded_pl = F.relu(self.pl_embedding(pl))
        embedded_ol = F.relu(self.pl_embedding(ol))
        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        # print(embedded_area.shape)
        # h_a = embedded_area + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num
        # h_s = embedded_shot + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad + embedded_player_area_num + embedded_opponent_area_num

        # h_a = self.convolution(h_a.unsqueeze(1)).squeeze(1)
        # h_s = self.convolution(h_s.unsqueeze(1)).squeeze(1)
        # split player

        h_a = embedded_area + embedded_player
        h_s = embedded_shot + embedded_player
        h_bh = embedded_ball_height + embedded_player
        h_ba = embedded_ball_aroundhead + embedded_player
        h_bb = embedded_ball_backhnad + embedded_player
        h_pan = embedded_player_area_num + embedded_player
        h_oan = embedded_opponent_area_num + embedded_player
        h_pl = embedded_pl + embedded_player
        h_ol = embedded_ol + embedded_player

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]
        h_bh_A = h_bh[:, ::2]
        h_bh_B = h_bh[:, 1::2]
        h_ba_A = h_ba[:, ::2]
        h_ba_B = h_ba[:, 1::2]
        h_bb_A = h_bb[:, ::2]
        h_bb_B = h_bb[:, 1::2]
        h_pan_A = h_pan[:, ::2]
        h_pan_B = h_pan[:, 1::2]
        h_oan_A = h_oan[:, ::2]
        h_oan_B = h_oan[:, 1::2]
        h_pl_A = h_pl[:, ::2]
        h_pl_B = h_pl[:, 1::2]
        h_ol_A = h_ol[:, ::2]
        h_ol_B = h_ol[:, 1::2]

        # local
        decode_output_area = self.dropout(self.position_embedding(h_a, mode='encode'))
        decode_output_shot = self.dropout(self.position_embedding(h_s, mode='encode'))
        decode_output_bh = self.dropout(self.position_embedding(h_bh, mode='encode'))
        decode_output_ba = self.dropout(self.position_embedding(h_ba, mode='encode'))
        decode_output_bb = self.dropout(self.position_embedding(h_bb, mode='encode'))
        decode_output_pan = self.dropout(self.position_embedding(h_pan, mode='encode'))
        decode_output_oan = self.dropout(self.position_embedding(h_oan, mode='encode'))
        decode_output_pl = self.dropout(self.position_embedding(h_pl, mode='encode'))
        decode_output_ol = self.dropout(self.position_embedding(h_ol, mode='encode'))
        # global
        decode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='encode'))
        decode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='encode'))
        decode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='encode'))
        decode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='encode'))
        decode_output_bh_A = self.dropout(self.position_embedding(h_bh_A, mode='encode'))
        decode_output_bh_B = self.dropout(self.position_embedding(h_bh_B, mode='encode'))
        decode_output_ba_A = self.dropout(self.position_embedding(h_ba_A, mode='encode'))
        decode_output_ba_B = self.dropout(self.position_embedding(h_ba_B, mode='encode'))
        decode_output_bb_A = self.dropout(self.position_embedding(h_bb_A, mode='encode'))
        decode_output_bb_B = self.dropout(self.position_embedding(h_bb_B, mode='encode'))
        decode_output_pan_A = self.dropout(self.position_embedding(h_pan_A, mode='encode'))
        decode_output_pan_B = self.dropout(self.position_embedding(h_pan_B, mode='encode'))
        decode_output_oan_A = self.dropout(self.position_embedding(h_oan_A, mode='encode'))
        decode_output_oan_B = self.dropout(self.position_embedding(h_oan_B, mode='encode'))
        decode_output_pl_A = self.dropout(self.position_embedding(h_pl_A, mode='encode'))
        decode_output_pl_B = self.dropout(self.position_embedding(h_pl_B, mode='encode'))
        decode_output_ol_A = self.dropout(self.position_embedding(h_ol_A, mode='encode'))
        decode_output_ol_B = self.dropout(self.position_embedding(h_ol_B, mode='encode'))

        for i in range(0, self.config['n_layers']):
            decode_output_area_A, decode_output_shot_A, decode_output_bh_A, decode_output_ba_A, decode_output_bb_A, decode_output_pan_A, decode_output_oan_A, decode_output_pl_A, decode_output_ol_A, dec_slf_attn_A, dec_enc_attn_A, disentangled_weight_A = self.global_layer_A(decode_output_area_A, decode_output_shot_A, decode_output_bh_A, decode_output_ba_A, decode_output_bb_A, decode_output_pan_A, decode_output_oan_A, decode_output_pl_A, decode_output_ol_A,
                                                                                                                                    encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A, slf_attn_mask=trg_global_A_mask, return_attns=return_attns)
            if decode_output_area_B.shape[1] != 0:
                decode_output_area_B, decode_output_shot_B, decode_output_bh_B, decode_output_ba_B, decode_output_bb_B, decode_output_pan_B, decode_output_oan_B, decode_output_pl_B, decode_output_ol_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer_B(decode_output_area_B, decode_output_shot_B, decode_output_bh_B, decode_output_ba_B, decode_output_bb_B, decode_output_pan_B, decode_output_oan_B, decode_output_pl_B, decode_output_ol_B,
                                                                                                                                    encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)
                # decode_output_area_B, decode_output_shot_B, dec_slf_attn_B, dec_enc_attn_B, disentangled_weight_B = self.global_layer_B(decode_output_area_B, decode_output_shot_B, encode_global_area_B, encode_global_shot_B, slf_attn_mask=trg_global_B_mask, return_attns=return_attns)

            decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decode_output_pl, decode_output_ol, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decode_output_pl, decode_output_ol,
                                                                                                                                    encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol, slf_attn_mask=trg_local_mask, return_attns=return_attns)
            # decode_output_area, decode_output_shot, dec_slf_attn, dec_enc_attn, disentangled_weight_local = self.local_layer(decode_output_area, decode_output_shot, encode_local_output_area, encode_local_output_shot, slf_attn_mask=trg_local_mask, return_attns=return_attns)
            
        decoder_self_attention_list = dec_slf_attn if return_attns else []
        decoder_encoder_self_attention_list = dec_enc_attn if return_attns else []

        if decode_output_area_B.shape[1] != 0:
            decode_output_A_area = alternatemerge(decode_output_area_A, decode_output_area_A, decode_output_area.shape[1], 'A')
            decode_output_A_shot = alternatemerge(decode_output_shot_A, decode_output_shot_A, decode_output_shot.shape[1], 'A')
            decode_output_A_bh = alternatemerge(decode_output_bh_A, decode_output_bh_A, decode_output_bh.shape[1], 'A')
            decode_output_A_ba = alternatemerge(decode_output_ba_A, decode_output_ba_A, decode_output_ba.shape[1], 'A')
            decode_output_A_bb = alternatemerge(decode_output_bb_A, decode_output_bb_A, decode_output_bb.shape[1], 'A')
            decode_output_A_pan = alternatemerge(decode_output_pan_A, decode_output_pan_A, decode_output_pan.shape[1], 'A')
            decode_output_A_oan = alternatemerge(decode_output_oan_A, decode_output_oan_A, decode_output_oan.shape[1], 'A')
            decode_output_A_pl = alternatemerge(decode_output_pl_A, decode_output_pl_A, decode_output_pl.shape[1], 'A')
            decode_output_A_ol = alternatemerge(decode_output_ol_A, decode_output_ol_A, decode_output_ol.shape[1], 'A')
            decode_output_B_area = alternatemerge(decode_output_area_B, decode_output_area_B, decode_output_area.shape[1], 'B')
            decode_output_B_shot = alternatemerge(decode_output_shot_B, decode_output_shot_B, decode_output_shot.shape[1], 'B')
            decode_output_B_bh = alternatemerge(decode_output_bh_B, decode_output_bh_B, decode_output_bh.shape[1], 'B')
            decode_output_B_ba = alternatemerge(decode_output_ba_B, decode_output_ba_B, decode_output_ba.shape[1], 'B')
            decode_output_B_bb = alternatemerge(decode_output_bb_B, decode_output_bb_B, decode_output_bb.shape[1], 'B')
            decode_output_B_pan = alternatemerge(decode_output_pan_B, decode_output_pan_B, decode_output_pan.shape[1], 'B')
            decode_output_B_oan = alternatemerge(decode_output_oan_B, decode_output_oan_B, decode_output_oan.shape[1], 'B')
            decode_output_B_pl = alternatemerge(decode_output_pl_B, decode_output_pl_B, decode_output_pl.shape[1], 'B')
            decode_output_B_ol = alternatemerge(decode_output_ol_B, decode_output_ol_B, decode_output_ol.shape[1], 'B')
        else:
            decode_output_A_area = decode_output_area_A.clone()
            decode_output_A_shot = decode_output_shot_A.clone()
            decode_output_A_bh = decode_output_bh_A.clone()
            decode_output_A_ba = decode_output_ba_A.clone()
            decode_output_A_bb = decode_output_bb_A.clone()
            decode_output_A_pan = decode_output_pan_A.clone()
            decode_output_A_oan = decode_output_oan_A.clone()
            decode_output_A_pl = decode_output_pl_A.clone()
            decode_output_A_ol = decode_output_ol_A.clone()
            decode_output_B_area = torch.zeros(decode_output_area.shape, device=decode_output_area.device)
            decode_output_B_shot = torch.zeros(decode_output_shot.shape, device=decode_output_shot.device)
            decode_output_B_bh = torch.zeros(decode_output_bh.shape, device=decode_output_bh.device)
            decode_output_B_ba = torch.zeros(decode_output_ba.shape, device=decode_output_ba.device)
            decode_output_B_bb = torch.zeros(decode_output_bb.shape, device=decode_output_bb.device)
            decode_output_B_pan = torch.zeros(decode_output_pan.shape, device=decode_output_pan.device)
            decode_output_B_oan = torch.zeros(decode_output_oan.shape, device=decode_output_oan.device)
            decode_output_B_pl = torch.zeros(decode_output_pl.shape, device=decode_output_pl.device)
            decode_output_B_ol = torch.zeros(decode_output_ol.shape, device=decode_output_ol.device)

        decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decode_output_pl, decode_output_ol = self.gated_fusion(decode_output_A_area, decode_output_B_area, decode_output_area, decode_output_A_shot, decode_output_B_shot, decode_output_shot,
                                                                   decode_output_A_bh, decode_output_B_bh, decode_output_bh, decode_output_A_ba, decode_output_B_ba, decode_output_ba,
                                                                   decode_output_A_bb, decode_output_B_bb, decode_output_bb, decode_output_A_pan, decode_output_B_pan, decode_output_pan,
                                                                   decode_output_A_oan, decode_output_B_oan, decode_output_oan, decode_output_A_pl, decode_output_B_pl, decode_output_pl,
                                                                   decode_output_A_ol, decode_output_B_ol, decode_output_ol)

        # (batch, seq_len, encode_dim)
        if return_attns:
            return decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decode_output_pl, decode_output_ol, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        return decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decode_output_pl, decode_output_ol


class ShotGenPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, return_attns=return_attns)
        else:
            decode_output = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output, encode_global_A, encode_global_B, return_attns)
       

        decode_output = (decode_output + embedded_target_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)
        height_logits = self.height_decoder(decode_output)
        aroundhead_logits = self.aroundhead_decoder(decode_output)
        backhand_logits = self.backhand_decoder(decode_output)
        playerloc_logits = self.location_decoder(decode_output)
        opponentloc_logits = self.location_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits
class ShotGenPredictor2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder2(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
        else:
            decode_output = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
       

        decode_output = (decode_output + embedded_target_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)
        height_logits = self.height_decoder(decode_output)
        aroundhead_logits = self.aroundhead_decoder(decode_output)
        backhand_logits = self.backhand_decoder(decode_output)
        playerloc_logits = self.location_decoder(decode_output)
        opponentloc_logits = self.location_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits
class ShotGenPredictor_model3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder_model3(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output_area, decode_output_shot, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
        else:
            decode_output_area, decode_output_shot = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
       

        decode_output_area = (decode_output_area + embedded_target_player)
        decode_output_shot = (decode_output_shot + embedded_target_player)
        decode_output = (decode_output_area + decode_output_shot + embedded_target_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)
        height_logits = self.height_decoder(decode_output)
        aroundhead_logits = self.aroundhead_decoder(decode_output)
        backhand_logits = self.backhand_decoder(decode_output)
        playerloc_logits = self.location_decoder(decode_output)
        opponentloc_logits = self.location_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits
class ShotGenPredictor_model_hybrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder_shot = ShotGenDecoder_model3(config)
        self.shotgen_decoder_area = ShotGenDecoder(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encoder_result, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)

        decode_output_area, decode_output_shot = self.shotgen_decoder_shot(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
            input_player_location_area, input_opponent_location_area, encoder_result['shot'][0], encoder_result['shot'][1], encoder_result['shot'][2], encoder_result['shot'][3], encoder_result['shot'][4], encoder_result['shot'][5], return_attns=return_attns)
       

        # decode_output_area = (decode_output_area + embedded_target_player)
        # decode_output_shot = (decode_output_shot + embedded_target_player)
        decode_output = (decode_output_area + decode_output_shot + embedded_target_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)
        height_logits = self.height_decoder(decode_output)
        aroundhead_logits = self.aroundhead_decoder(decode_output)
        backhand_logits = self.backhand_decoder(decode_output)
        playerloc_logits = self.location_decoder(decode_output)
        opponentloc_logits = self.location_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits
class ShotGenPredictor_model3_ver2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder_model3_ver2(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output_area, decode_output_shot, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
        else:
            decode_output_area, decode_output_shot = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
       

        decode_output_area = (decode_output_area + embedded_target_player)
        decode_output_shot = (decode_output_shot + embedded_target_player)
        decode_output = (decode_output_area + decode_output_shot + embedded_target_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)
        height_logits = self.height_decoder(decode_output)
        aroundhead_logits = self.aroundhead_decoder(decode_output)
        backhand_logits = self.backhand_decoder(decode_output)
        playerloc_logits = self.location_decoder(decode_output)
        opponentloc_logits = self.location_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits

class ShotGenPredictor_model3_ver3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder_model3_ver3(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
        else:
            decode_output = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
    

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)
        height_logits = self.height_decoder(decode_output)
        aroundhead_logits = self.aroundhead_decoder(decode_output)
        backhand_logits = self.backhand_decoder(decode_output)
        playerloc_logits = self.location_decoder(decode_output)
        opponentloc_logits = self.location_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits
class ShotGenPredictor_model3_small(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder_model3_small(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output_area, decode_output_shot, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
        else:
            decode_output_area, decode_output_shot = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
       

        decode_output_area = (decode_output_area + embedded_target_player)
        decode_output_shot = (decode_output_shot + embedded_target_player)
        decode_output = (decode_output_area + decode_output_shot + embedded_target_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)
        # height_logits = self.height_decoder(decode_output)
        aroundhead_logits = self.aroundhead_decoder(decode_output)
        backhand_logits = self.backhand_decoder(decode_output)
        # playerloc_logits = self.location_decoder(decode_output)
        # opponentloc_logits = self.location_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, aroundhead_logits, backhand_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, aroundhead_logits, backhand_logits
class ShotGenPredictor4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder_model3(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output_area, decode_output_shot, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
        else:
            decode_output_area, decode_output_shot = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
       

        decode_output_area = (decode_output_area + embedded_target_player)
        decode_output_shot = (decode_output_shot + embedded_target_player)
        decode_output = (decode_output_area + decode_output_shot + embedded_target_player)

        area_logits = self.area_decoder(decode_output_area)
        shot_logits = self.shot_decoder(decode_output_shot)
        height_logits = self.height_decoder(decode_output)
        aroundhead_logits = self.aroundhead_decoder(decode_output)
        backhand_logits = self.backhand_decoder(decode_output)
        playerloc_logits = self.location_decoder(decode_output)
        opponentloc_logits = self.location_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits
class ShotGenPredictorSimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoderSimple(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output_area, decode_output_shot, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
        else:
            decode_output_area, decode_output_shot = self.shotgen_decoder(input_shot, input_x, input_y, input_player, encode_local_output_area, encode_local_output_shot, 
                encode_global_area_A, encode_global_shot_A, encode_global_area_B, encode_global_shot_B, return_attns=return_attns)
       

        decode_output_area = (decode_output_area + embedded_target_player)
        decode_output_shot = (decode_output_shot + embedded_target_player)
        decode_output = (decode_output_area + decode_output_shot + embedded_target_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits
class ShotGenPredictor_Big(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder_Big(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, return_attns=return_attns)
        else:
            decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, return_attns=return_attns)
       

        decode_output_area = (decode_output_area + embedded_target_player)
        decode_output_shot = (decode_output_shot + embedded_target_player)
        decode_output_bh = (decode_output_bh + embedded_target_player)
        decode_output_ba = (decode_output_ba + embedded_target_player)
        decode_output_bb = (decode_output_bb + embedded_target_player)
        decode_output_pan = (decode_output_pan + embedded_target_player)
        decode_output_oan = (decode_output_oan + embedded_target_player)
        decode_output = (decode_output_area + decode_output_shot + embedded_target_player)

        area_logits = self.area_decoder(decode_output_area)
        shot_logits = self.shot_decoder(decode_output_shot)
        height_logits = self.height_decoder(decode_output_bh)
        aroundhead_logits = self.aroundhead_decoder(decode_output_ba)
        backhand_logits = self.backhand_decoder(decode_output_bb)
        playerloc_logits = self.location_decoder(decode_output_pan)
        opponentloc_logits = self.location_decoder(decode_output_oan)

        if return_attns:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits
class ShotGenPredictor_Big2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder_Big2(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.pl_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.ol_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, input_player_location_x, input_player_location_y, input_opponent_location_x, input_opponent_location_y,
                  encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol,
                  encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A,
                  encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B,
                  target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decode_output_pl, decode_output_ol, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, input_player_location_x, input_player_location_y, input_opponent_location_x, input_opponent_location_y, 
                encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol,
                encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A,
                encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B,
                return_attns=return_attns)
        else:
            decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decode_output_pl, decode_output_ol = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, input_player_location_x, input_player_location_y, input_opponent_location_x, input_opponent_location_y, 
                encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol,
                encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A,
                encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B,
                return_attns=return_attns)

        decode_output_area = (decode_output_area + embedded_target_player)
        decode_output_shot = (decode_output_shot + embedded_target_player)
        decode_output_bh = (decode_output_bh + embedded_target_player)
        decode_output_ba = (decode_output_ba + embedded_target_player)
        decode_output_bb = (decode_output_bb + embedded_target_player)
        decode_output_pan = (decode_output_pan + embedded_target_player)
        decode_output_oan = (decode_output_oan + embedded_target_player)
        decode_output_pl = (decode_output_pl + embedded_target_player)
        decode_output_ol = (decode_output_ol + embedded_target_player)
        # print(decode_output_area.shape, decode_output_pl.shape, decode_output_ol.shape)
        decode_output = (decode_output_area + decode_output_shot + embedded_target_player)

        area_logits = self.area_decoder(decode_output_area)
        shot_logits = self.shot_decoder(decode_output_shot)
        height_logits = self.height_decoder(decode_output_bh)
        aroundhead_logits = self.aroundhead_decoder(decode_output_ba)
        backhand_logits = self.backhand_decoder(decode_output_bb)
        playerloc_logits = self.location_decoder(decode_output_pan)
        opponentloc_logits = self.location_decoder(decode_output_oan)
        pl_logits = self.pl_decoder(decode_output_pl)
        ol_logits = self.ol_decoder(decode_output_ol)
        if return_attns:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, pl_logits, ol_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, pl_logits, ol_logits
class ShotGenPredictor_Big3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder_Big(config)
        self.area_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        )
        self.shot_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], config['shot_num'], bias=False)
        )
        self.aroundhead_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.backhand_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.height_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 3, bias=False)
        )
        self.location_decoder = nn.Sequential(
            nn.Linear(config['encode_dim'], 11, bias=False)
        )
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, target_player, return_attns=False):
        
        embedded_target_player = self.player_embedding(target_player)
        if return_attns:
            decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, return_attns=return_attns)
        else:
            decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan = self.shotgen_decoder(input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand,
                input_player_location_area, input_opponent_location_area, encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, return_attns=return_attns)
       

        decode_output_area = (decode_output_area + embedded_target_player)
        decode_output_shot = (decode_output_shot + embedded_target_player)
        decode_output_bh = (decode_output_bh + embedded_target_player)
        decode_output_ba = (decode_output_ba + embedded_target_player)
        decode_output_bb = (decode_output_bb + embedded_target_player)
        decode_output_pan = (decode_output_pan + embedded_target_player)
        decode_output_oan = (decode_output_oan + embedded_target_player)
        decode_output = (decode_output_area + decode_output_shot + decode_output_bh + decode_output_ba + decode_output_bb + decode_output_pan + decode_output_oan + embedded_target_player)

        area_logits = self.area_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)
        height_logits = self.height_decoder(decode_output)
        aroundhead_logits = self.aroundhead_decoder(decode_output)
        backhand_logits = self.backhand_decoder(decode_output)
        playerloc_logits = self.location_decoder(decode_output)
        opponentloc_logits = self.location_decoder(decode_output)

        if return_attns:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits, decoder_self_attention_list, decoder_encoder_self_attention_list, disentangled_weight_local
        else:
            return area_logits, shot_logits, height_logits, aroundhead_logits, backhand_logits, playerloc_logits, opponentloc_logits


class ShotGenEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer = EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                input_player_location_area, input_opponent_location_area, src_mask=None, return_attns=False):
        
        enc_slf_attn_list = []

        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)
        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        h_a = embedded_area + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad +  embedded_player_area_num + embedded_opponent_area_num
        h_s = embedded_shot + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad +  embedded_player_area_num + embedded_opponent_area_num

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        encode_output_area = self.dropout(self.position_embedding(h_a, mode='encode'))
        encode_output_shot = self.dropout(self.position_embedding(h_s, mode='encode'))
        # global
        encode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='encode'))
        encode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='encode'))
        encode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='encode'))
        encode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='encode'))

        encode_global_A, enc_slf_attn_A = self.global_layer(encode_output_area_A, encode_output_shot_A, slf_attn_mask=src_mask)
        encode_global_B, enc_slf_attn_B = self.global_layer(encode_output_area_B, encode_output_shot_B, slf_attn_mask=src_mask)
        
        encode_local_output, enc_slf_attn = self.local_layer(encode_output_area, encode_output_shot, slf_attn_mask=src_mask)

        if return_attns:
            return encode_local_output, encode_global_A, encode_global_B, enc_slf_attn_list
        return encode_local_output, encode_global_A, encode_global_B
class ShotGenEncoder_model3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                input_player_location_area, input_opponent_location_area, src_mask=None, return_attns=False):
        
        enc_slf_attn_list = []

        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)
        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        h_a = embedded_area + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad +  embedded_player_area_num + embedded_opponent_area_num
        h_s = embedded_shot + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad +  embedded_player_area_num + embedded_opponent_area_num

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        encode_output_area = self.dropout(self.position_embedding(h_a, mode='encode'))
        encode_output_shot = self.dropout(self.position_embedding(h_s, mode='encode'))
        # global
        encode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='encode'))
        encode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='encode'))
        encode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='encode'))
        encode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='encode'))

        for i in range(0, self.config['n_layers']):
            encode_output_area_A, encode_output_shot_A, enc_slf_attn_A = self.global_layer_A(encode_output_area_A, encode_output_shot_A, slf_attn_mask=src_mask)
            encode_output_area_B, encode_output_shot_B, enc_slf_attn_B = self.global_layer_B(encode_output_area_B, encode_output_shot_B, slf_attn_mask=src_mask)
            encode_output_area, encode_output_shot, enc_slf_attn = self.local_layer(encode_output_area, encode_output_shot, slf_attn_mask=src_mask)

        if return_attns:
            return encode_output_area, encode_output_shot, encode_output_area_A, encode_output_shot_A, encode_output_area_B, encode_output_shot_B, enc_slf_attn_list
        return encode_output_area, encode_output_shot, encode_output_area_A, encode_output_shot_A, encode_output_area_B, encode_output_shot_B
class ShotGenEncoder_model3_ver2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                input_player_location_area, input_opponent_location_area, src_mask=None, return_attns=False):
        
        enc_slf_attn_list = []

        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)
        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        h_a = embedded_area + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad +  embedded_player_area_num + embedded_opponent_area_num
        h_s = embedded_shot + embedded_player + embedded_ball_height + embedded_ball_aroundhead + embedded_ball_backhnad +  embedded_player_area_num + embedded_opponent_area_num

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        encode_output_area = self.dropout(self.position_embedding(h_a, mode='encode'))
        encode_output_shot = self.dropout(self.position_embedding(h_s, mode='encode'))
        # global
        encode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='encode'))
        encode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='encode'))
        encode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='encode'))
        encode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='encode'))

        for i in range(0, self.config['n_layers']):
            encode_output_area_A, encode_output_shot_A, enc_slf_attn_A = self.global_layer(encode_output_area_A, encode_output_shot_A, slf_attn_mask=src_mask)
            encode_output_area_B, encode_output_shot_B, enc_slf_attn_B = self.global_layer(encode_output_area_B, encode_output_shot_B, slf_attn_mask=src_mask)
            encode_output_area, encode_output_shot, enc_slf_attn = self.local_layer(encode_output_area, encode_output_shot, slf_attn_mask=src_mask)

        if return_attns:
            return encode_output_area, encode_output_shot, encode_output_area_A, encode_output_shot_A, encode_output_area_B, encode_output_shot_B, enc_slf_attn_list
        return encode_output_area, encode_output_shot, encode_output_area_A, encode_output_shot_A, encode_output_area_B, encode_output_shot_B
class ShotGenEncoder_hybrid(nn.Module):
    def __init__(self, config):
        self.encoder_shot = ShotGenEncoder_model3(config)
        self.encoder_area = ShotGenEncoder(config)
    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                input_player_location_area, input_opponent_location_area, src_mask=None, return_attns=False):
        encode_local_output, encode_global_A, encode_global_B = self.encoder_area(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                input_player_location_area, input_opponent_location_area, src_mask, return_attns)
        encode_output_area, encode_output_shot, encode_output_area_A, encode_output_shot_A, encode_output_area_B, encode_output_shot_B = self.encoder_shot(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                input_player_location_area, input_opponent_location_area, src_mask, return_attns)
        result = {
            'area': (encode_local_output, encode_global_A, encode_global_B),
            'shot': (encode_output_area, encode_output_shot, encode_output_area_A, encode_output_shot_A, encode_output_area_B, encode_output_shot_B)
        }
        return result

class ShotGenEncoder_model3_small(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

    def forward(self, input_shot, input_x, input_y, input_player, input_aroundhead, input_backhand, src_mask=None, return_attns=False):
        
        enc_slf_attn_list = []

        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)
        # embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        # embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        # embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        h_a = embedded_area + embedded_player + embedded_ball_aroundhead + embedded_ball_backhnad
        h_s = embedded_shot + embedded_player + embedded_ball_aroundhead + embedded_ball_backhnad

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        encode_output_area = self.dropout(self.position_embedding(h_a, mode='encode'))
        encode_output_shot = self.dropout(self.position_embedding(h_s, mode='encode'))
        # global
        encode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='encode'))
        encode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='encode'))
        encode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='encode'))
        encode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='encode'))

        for i in range(0, self.config['n_layers']):
            encode_output_area_A, encode_output_shot_A, enc_slf_attn_A = self.global_layer_A(encode_output_area_A, encode_output_shot_A, slf_attn_mask=src_mask)
            encode_output_area_B, encode_output_shot_B, enc_slf_attn_B = self.global_layer_B(encode_output_area_B, encode_output_shot_B, slf_attn_mask=src_mask)
            encode_output_area, encode_output_shot, enc_slf_attn = self.local_layer(encode_output_area, encode_output_shot, slf_attn_mask=src_mask)

        if return_attns:
            return encode_output_area, encode_output_shot, encode_output_area_A, encode_output_shot_A, encode_output_area_B, encode_output_shot_B, enc_slf_attn_list
        return encode_output_area, encode_output_shot, encode_output_area_A, encode_output_shot_A, encode_output_area_B, encode_output_shot_B  
class ShotGenEncoderSimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = EncoderLayer2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

    def forward(self, input_shot, input_x, input_y, input_player, src_mask=None, return_attns=False):
        
        enc_slf_attn_list = []

        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)

        h_a = embedded_area + embedded_player
        h_s = embedded_shot + embedded_player

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]

        # local
        encode_output_area = self.dropout(self.position_embedding(h_a, mode='encode'))
        encode_output_shot = self.dropout(self.position_embedding(h_s, mode='encode'))
        # global
        encode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='encode'))
        encode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='encode'))
        encode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='encode'))
        encode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='encode'))

        for i in range(0, self.config['n_layers']):
            encode_output_area_A, encode_output_shot_A, enc_slf_attn_A = self.global_layer_A(encode_output_area_A, encode_output_shot_A, slf_attn_mask=src_mask)
            encode_output_area_B, encode_output_shot_B, enc_slf_attn_B = self.global_layer_B(encode_output_area_B, encode_output_shot_B, slf_attn_mask=src_mask)
            encode_output_area, encode_output_shot, enc_slf_attn = self.local_layer(encode_output_area, encode_output_shot, slf_attn_mask=src_mask)

        if return_attns:
            return encode_output_area, encode_output_shot, encode_output_area_A, encode_output_shot_A, encode_output_area_B, encode_output_shot_B, enc_slf_attn_list
        return encode_output_area, encode_output_shot, encode_output_area_A, encode_output_shot_A, encode_output_area_B, encode_output_shot_B
class ShotGenEncoder_Big(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = EncoderLayer_Big(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = EncoderLayer_Big(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = EncoderLayer_Big(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                input_player_location_area, input_opponent_location_area, src_mask=None, return_attns=False):
        
        enc_slf_attn_list = []

        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)
        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        h_a = embedded_area + embedded_player
        h_s = embedded_shot + embedded_player
        h_bh = embedded_ball_height + embedded_player
        h_ba = embedded_ball_aroundhead + embedded_player
        h_bb = embedded_ball_backhnad + embedded_player
        h_pan = embedded_player_area_num + embedded_player
        h_oan = embedded_opponent_area_num + embedded_player

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]
        h_bh_A = h_bh[:, ::2]
        h_bh_B = h_bh[:, 1::2]
        h_ba_A = h_ba[:, ::2]
        h_ba_B = h_ba[:, 1::2]
        h_bb_A = h_bb[:, ::2]
        h_bb_B = h_bb[:, 1::2]
        h_pan_A = h_pan[:, ::2]
        h_pan_B = h_pan[:, 1::2]
        h_oan_A = h_oan[:, ::2]
        h_oan_B = h_oan[:, 1::2]

        # local
        encode_output_area = self.dropout(self.position_embedding(h_a, mode='encode'))
        encode_output_shot = self.dropout(self.position_embedding(h_s, mode='encode'))
        encode_output_bh = self.dropout(self.position_embedding(h_bh, mode='encode'))
        encode_output_ba = self.dropout(self.position_embedding(h_ba, mode='encode'))
        encode_output_bb = self.dropout(self.position_embedding(h_bb, mode='encode'))
        encode_output_pan = self.dropout(self.position_embedding(h_pan, mode='encode'))
        encode_output_oan = self.dropout(self.position_embedding(h_oan, mode='encode'))
        # global
        encode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='encode'))
        encode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='encode'))
        encode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='encode'))
        encode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='encode'))
        encode_output_bh_A = self.dropout(self.position_embedding(h_bh_A, mode='encode'))
        encode_output_bh_B = self.dropout(self.position_embedding(h_bh_B, mode='encode'))
        encode_output_ba_A = self.dropout(self.position_embedding(h_ba_A, mode='encode'))
        encode_output_ba_B = self.dropout(self.position_embedding(h_ba_B, mode='encode'))
        encode_output_bb_A = self.dropout(self.position_embedding(h_bb_A, mode='encode'))
        encode_output_bb_B = self.dropout(self.position_embedding(h_bb_B, mode='encode'))
        encode_output_pan_A = self.dropout(self.position_embedding(h_pan_A, mode='encode'))
        encode_output_pan_B = self.dropout(self.position_embedding(h_pan_B, mode='encode'))
        encode_output_oan_A = self.dropout(self.position_embedding(h_oan_A, mode='encode'))
        encode_output_oan_B = self.dropout(self.position_embedding(h_oan_B, mode='encode'))



        for i in range(0, self.config['n_layers']):
            encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, enc_slf_attn_A = self.global_layer_A(encode_output_area_A, encode_output_shot_A, 
                                                                                             encode_output_bh_A, encode_output_ba_A,
                                                                                             encode_output_bb_A, encode_output_pan_A,
                                                                                             encode_output_oan_A,slf_attn_mask=src_mask)
            encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, enc_slf_attn_B = self.global_layer_B(encode_output_area_B, encode_output_shot_B, 
                                                                                             encode_output_bh_B, encode_output_ba_B,
                                                                                             encode_output_bb_B, encode_output_pan_B,
                                                                                             encode_output_oan_B,slf_attn_mask=src_mask)
            # encode_output_area_B, encode_output_shot_B, enc_slf_attn_B = self.global_layer_B(encode_output_area_B, encode_output_shot_B, slf_attn_mask=src_mask)
            # encode_output_area, encode_output_shot, enc_slf_attn = self.local_layer(encode_output_area, encode_output_shot, slf_attn_mask=src_mask)
            encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, enc_slf_attn = self.local_layer(encode_output_area, encode_output_shot, 
                                                                                             encode_output_bh, encode_output_ba,
                                                                                             encode_output_bb, encode_output_pan,
                                                                                             encode_output_oan,slf_attn_mask=src_mask)

        if return_attns:
            return encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, enc_slf_attn_list
        return encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B
class ShotGenEncoder_Big2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.area_pl_embedding = nn.Linear(2, config['area_dim'])
        self.area_ol_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = ShotEmbedding(config['shot_num'], config['shot_dim'])
        self.player_embedding = PlayerEmbedding(config['player_num'], config['player_dim'])
        self.bool_embedding = BoolEmbedding(3, config['player_dim'])
        self.areaNum_embedding = AreaEmbedding(11, config['player_dim'])
        self.score_embedding = AreaEmbedding(28, config['player_dim'])

        n_heads = 2
        d_k = config['encode_dim']
        d_v = config['encode_dim']
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = 0.1
        self.d_model = d_model

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=config['max_ball_round'])
        self.dropout = nn.Dropout(p=dropout)

        self.global_layer_A = EncoderLayer_Big2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = EncoderLayer_Big2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = EncoderLayer_Big2(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                input_player_location_area, input_opponent_location_area,
                input_player_location_x, input_player_location_y, input_opponent_location_x, input_opponent_location_y, src_mask=None, return_attns=False):
        
        enc_slf_attn_list = []

        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        embedded_area = F.relu(self.area_embedding(area))
        area_pl = torch.cat((input_player_location_x.unsqueeze(-1), input_player_location_y.unsqueeze(-1)), dim=-1).float()
        embedded_area_pl = F.relu(self.area_pl_embedding(area_pl))
        area_ol = torch.cat((input_opponent_location_x.unsqueeze(-1), input_opponent_location_y.unsqueeze(-1)), dim=-1).float()
        embedded_area_ol = F.relu(self.area_ol_embedding(area_ol))

        embedded_shot = self.shot_embedding(input_shot)
        embedded_player = self.player_embedding(input_player)
        embedded_ball_height = self.bool_embedding(input_height)
        embedded_ball_aroundhead = self.bool_embedding(input_aroundhead)
        embedded_ball_backhnad = self.bool_embedding(input_backhand)
        embedded_player_area_num = self.areaNum_embedding(input_player_location_area)
        embedded_opponent_area_num = self.areaNum_embedding(input_opponent_location_area)

        h_a = embedded_area + embedded_player
        h_pla = embedded_area_pl + embedded_player
        h_ola = embedded_area_ol + embedded_player
        h_s = embedded_shot + embedded_player
        h_bh = embedded_ball_height + embedded_player
        h_ba = embedded_ball_aroundhead + embedded_player
        h_bb = embedded_ball_backhnad + embedded_player
        h_pan = embedded_player_area_num + embedded_player
        h_oan = embedded_opponent_area_num + embedded_player

        # split player
        h_a_A = h_a[:, ::2]
        h_a_B = h_a[:, 1::2]
        h_pla_A = h_pla[:, ::2]
        h_pla_B = h_pla[:, 1::2]
        h_ola_A = h_ola[:, ::2]
        h_ola_B = h_ola[:, 1::2]
        h_s_A = h_s[:, ::2]
        h_s_B = h_s[:, 1::2]
        h_bh_A = h_bh[:, ::2]
        h_bh_B = h_bh[:, 1::2]
        h_ba_A = h_ba[:, ::2]
        h_ba_B = h_ba[:, 1::2]
        h_bb_A = h_bb[:, ::2]
        h_bb_B = h_bb[:, 1::2]
        h_pan_A = h_pan[:, ::2]
        h_pan_B = h_pan[:, 1::2]
        h_oan_A = h_oan[:, ::2]
        h_oan_B = h_oan[:, 1::2]

        # local
        encode_output_area = self.dropout(self.position_embedding(h_a, mode='encode'))
        encode_output_pl = self.dropout(self.position_embedding(h_pla, mode='encode'))
        encode_output_ol = self.dropout(self.position_embedding(h_ola, mode='encode'))
        encode_output_shot = self.dropout(self.position_embedding(h_s, mode='encode'))
        encode_output_bh = self.dropout(self.position_embedding(h_bh, mode='encode'))
        encode_output_ba = self.dropout(self.position_embedding(h_ba, mode='encode'))
        encode_output_bb = self.dropout(self.position_embedding(h_bb, mode='encode'))
        encode_output_pan = self.dropout(self.position_embedding(h_pan, mode='encode'))
        encode_output_oan = self.dropout(self.position_embedding(h_oan, mode='encode'))
        # global
        encode_output_area_A = self.dropout(self.position_embedding(h_a_A, mode='encode'))
        encode_output_area_B = self.dropout(self.position_embedding(h_a_B, mode='encode'))
        encode_output_shot_A = self.dropout(self.position_embedding(h_s_A, mode='encode'))
        encode_output_shot_B = self.dropout(self.position_embedding(h_s_B, mode='encode'))
        encode_output_bh_A = self.dropout(self.position_embedding(h_bh_A, mode='encode'))
        encode_output_bh_B = self.dropout(self.position_embedding(h_bh_B, mode='encode'))
        encode_output_ba_A = self.dropout(self.position_embedding(h_ba_A, mode='encode'))
        encode_output_ba_B = self.dropout(self.position_embedding(h_ba_B, mode='encode'))
        encode_output_bb_A = self.dropout(self.position_embedding(h_bb_A, mode='encode'))
        encode_output_bb_B = self.dropout(self.position_embedding(h_bb_B, mode='encode'))
        encode_output_pan_A = self.dropout(self.position_embedding(h_pan_A, mode='encode'))
        encode_output_pan_B = self.dropout(self.position_embedding(h_pan_B, mode='encode'))
        encode_output_oan_A = self.dropout(self.position_embedding(h_oan_A, mode='encode'))
        encode_output_oan_B = self.dropout(self.position_embedding(h_oan_B, mode='encode'))
        encode_output_pl_A = self.dropout(self.position_embedding(h_pla_A, mode='encode'))
        encode_output_pl_B = self.dropout(self.position_embedding(h_pla_B, mode='encode'))
        encode_output_ol_A = self.dropout(self.position_embedding(h_ola_A, mode='encode'))
        encode_output_ol_B = self.dropout(self.position_embedding(h_ola_B, mode='encode'))



        for i in range(0, self.config['n_layers']):
            encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A, enc_slf_attn_A = self.global_layer_A(encode_output_area_A, encode_output_shot_A, 
                                                                                             encode_output_bh_A, encode_output_ba_A,
                                                                                             encode_output_bb_A, encode_output_pan_A,
                                                                                             encode_output_oan_A, encode_output_pl_A, encode_output_ol_A, slf_attn_mask=src_mask)
            encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B, enc_slf_attn_B = self.global_layer_B(encode_output_area_B, encode_output_shot_B, 
                                                                                             encode_output_bh_B, encode_output_ba_B,
                                                                                             encode_output_bb_B, encode_output_pan_B,
                                                                                             encode_output_oan_B, encode_output_pl_B, encode_output_ol_B, slf_attn_mask=src_mask)
            # encode_output_area_B, encode_output_shot_B, enc_slf_attn_B = self.global_layer_B(encode_output_area_B, encode_output_shot_B, slf_attn_mask=src_mask)
            # encode_output_area, encode_output_shot, enc_slf_attn = self.local_layer(encode_output_area, encode_output_shot, slf_attn_mask=src_mask)
            encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol, enc_slf_attn = self.local_layer(encode_output_area, encode_output_shot, 
                                                                                             encode_output_bh, encode_output_ba,
                                                                                             encode_output_bb, encode_output_pan,
                                                                                             encode_output_oan, encode_output_pl, encode_output_ol, slf_attn_mask=src_mask)

        if return_attns:
            return encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B, enc_slf_attn_list
        return encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol, encode_output_area_A, encode_output_shot_A, encode_output_bh_A, encode_output_ba_A, encode_output_bb_A, encode_output_pan_A, encode_output_oan_A, encode_output_pl_A, encode_output_ol_A, encode_output_area_B, encode_output_shot_B, encode_output_bh_B, encode_output_ba_B, encode_output_bb_B, encode_output_pan_B, encode_output_oan_B, encode_output_pl_B, encode_output_ol_B