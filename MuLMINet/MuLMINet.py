import torch
import torch.nn as nn
import torch.nn.functional as F
from MuLMINet.MuLMINet_layers import EncoderLayer_MuLMINet, DecoderLayer_MuLMINet, GatedFusionLayer_MuLMINet_Variant2, GatedFusionLayer_MuLMINet_Variant1
from MuLMINet.MuLMINet_embedding import PositionalEncoding, PlayerEmbedding, ShotEmbedding, BoolEmbedding, AreaEmbedding

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

class ShotGenDecoder_MuLMINet_Variant1(nn.Module):
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

        self.global_layer_A = DecoderLayer_MuLMINet(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = DecoderLayer_MuLMINet(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer_MuLMINet(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer_MuLMINet_Variant1(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

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

class ShotGenDecoder_MuLMINet_Variant2(nn.Module):
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

        self.global_layer = DecoderLayer_MuLMINet(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = DecoderLayer_MuLMINet(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

        self.gated_fusion = GatedFusionLayer_MuLMINet_Variant2(d_model, d_model, config['encode_length'], config['max_ball_round']+1)

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

class ShotGenPredictor_MuLMINet_Variant1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder_MuLMINet_Variant1(config)
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

class ShotGenPredictor_MuLMINet_Variant2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder_MuLMINet_Variant2(config)
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

class ShotGenEncoder_MuLMINet_Variant1(nn.Module):
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

        self.global_layer_A = EncoderLayer_MuLMINet(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.global_layer_B = EncoderLayer_MuLMINet(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = EncoderLayer_MuLMINet(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                input_player_location_area, input_opponent_location_area, src_mask=None, return_attns=False):
        
        enc_slf_attn_list = []

        input_height[input_height == -2147483648] = 1 # to replace the overflow value

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

class ShotGenEncoder_MuLMINet_Variant2(nn.Module):
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

        self.global_layer = EncoderLayer_MuLMINet(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
        self.local_layer = EncoderLayer_MuLMINet(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)

    def forward(self, input_shot, input_x, input_y, input_player, input_height, input_aroundhead, input_backhand, 
                input_player_location_area, input_opponent_location_area, src_mask=None, return_attns=False):
        
        enc_slf_attn_list = []
        input_height[input_height == -2147483648] = 1  # to replace the overflow value
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
