import torch.nn as nn
import torch
from ShuttleNet.ShuttleNet_submodules import TypeAreaMultiHeadAttention, TypeAreaMultiHeadAttention2, TypeAreaMultiHeadAttention_Big, TypeAreaMultiHeadAttention_Big2, MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.disentangled_attention = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, encode_area, encode_shot, slf_attn_mask=None):
        encode_output, enc_slf_attn, enc_disentangled_weight = self.disentangled_attention(encode_area, encode_area, encode_area, encode_shot, encode_shot, encode_shot, mask=slf_attn_mask)
        encode_output = self.pos_ffn(encode_output)
        return encode_output, enc_slf_attn
class EncoderLayer2(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.disentangled_attention = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, encode_area, encode_shot, slf_attn_mask=None):
        encode_output, enc_slf_attn, enc_disentangled_weight = self.disentangled_attention(encode_area, encode_area, encode_area, encode_shot, encode_shot, encode_shot, mask=slf_attn_mask)
        encode_output_area = self.pos_ffn1(encode_output)
        encode_output_shot = self.pos_ffn2(encode_output)
        return encode_output_area, encode_output_shot, enc_slf_attn
class EncoderLayer_Big(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.disentangled_attention = TypeAreaMultiHeadAttention_Big(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_area = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_shot = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_bh = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_ba = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_bb = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_pan = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_oan = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, encode_area, encode_shot, encode_output_bh, encode_output_ba, 
                encode_output_bb, encode_output_pan, encode_output_oan, slf_attn_mask=None):
        encode_output, enc_slf_attn, enc_disentangled_weight = self.disentangled_attention(encode_area, encode_area, encode_area, encode_shot, encode_shot, encode_shot, 
                                                                                           encode_output_bh, encode_output_bh, encode_output_bh,
                                                                                           encode_output_ba, encode_output_ba, encode_output_ba,
                                                                                           encode_output_bb, encode_output_bb, encode_output_bb,
                                                                                           encode_output_pan, encode_output_pan, encode_output_pan,
                                                                                           encode_output_oan, encode_output_oan, encode_output_oan, mask=slf_attn_mask)
        encode_output_area = self.pos_ffn_area(encode_output)
        encode_output_shot = self.pos_ffn_shot(encode_output)
        encode_output_bh = self.pos_ffn_bh(encode_output)
        encode_output_ba = self.pos_ffn_ba(encode_output)
        encode_output_bb = self.pos_ffn_bb(encode_output)
        encode_output_pan = self.pos_ffn_pan(encode_output)
        encode_output_oan = self.pos_ffn_oan(encode_output)
        return encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, enc_slf_attn
class EncoderLayer_Big2(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.disentangled_attention = TypeAreaMultiHeadAttention_Big2(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_area = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_pl = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_ol = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_shot = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_bh = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_ba = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_bb = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_pan = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_oan = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, encode_area, encode_shot, encode_output_bh, encode_output_ba, 
                encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol, slf_attn_mask=None):
        encode_output, enc_slf_attn, enc_disentangled_weight = self.disentangled_attention(encode_area, encode_area, encode_area, encode_shot, encode_shot, encode_shot, 
                                                                                           encode_output_bh, encode_output_bh, encode_output_bh,
                                                                                           encode_output_ba, encode_output_ba, encode_output_ba,
                                                                                           encode_output_bb, encode_output_bb, encode_output_bb,
                                                                                           encode_output_pan, encode_output_pan, encode_output_pan,
                                                                                           encode_output_oan, encode_output_oan, encode_output_oan, 
                                                                                           encode_output_pl, encode_output_pl, encode_output_pl,
                                                                                           encode_output_ol, encode_output_ol, encode_output_ol, mask=slf_attn_mask)
        encode_output_area = self.pos_ffn_area(encode_output)
        encode_output_shot = self.pos_ffn_shot(encode_output)
        encode_output_bh = self.pos_ffn_bh(encode_output)
        encode_output_ba = self.pos_ffn_ba(encode_output)
        encode_output_bb = self.pos_ffn_bb(encode_output)
        encode_output_pan = self.pos_ffn_pan(encode_output)
        encode_output_oan = self.pos_ffn_oan(encode_output)
        encode_output_pl = self.pos_ffn_pl(encode_output)
        encode_output_ol = self.pos_ffn_ol(encode_output)
        return encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.decoder_attention = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, decode_area, decode_shot, encode_output, slf_attn_mask=None, dec_enc_attn_mask=None, return_attns=None):
        decode_output, dec_slf_attn, disentangled_weight = self.decoder_attention(decode_area, decode_area, decode_area, decode_shot, decode_shot, decode_shot, mask=slf_attn_mask, return_attns=return_attns)
        decode_output, dec_enc_slf_attn = self.decoder_encoder_attention(decode_output, encode_output, encode_output, mask=dec_enc_attn_mask)
        decode_output = self.pos_ffn(decode_output)
        return decode_output, dec_slf_attn, dec_enc_slf_attn, disentangled_weight
class DecoderLayer2(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.decoder_attention = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention1 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention2 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, decode_area, decode_shot, encode_output_area, encode_output_shot, slf_attn_mask=None, dec_enc_attn_mask=None, return_attns=None):
        decode_output, dec_slf_attn, disentangled_weight = self.decoder_attention(decode_area, decode_area, decode_area, decode_shot, decode_shot, decode_shot, mask=slf_attn_mask, return_attns=return_attns)
        decode_output_area, dec_enc_slf_attn = self.decoder_encoder_attention1(decode_output, encode_output_area, encode_output_area, mask=dec_enc_attn_mask)
        decode_output_shot, dec_enc_slf_attn = self.decoder_encoder_attention2(decode_output, encode_output_shot, encode_output_shot, mask=dec_enc_attn_mask)
        decode_output_area = self.pos_ffn1(decode_output_area)
        decode_output_shot = self.pos_ffn1(decode_output_shot)
        return decode_output_area, decode_output_shot, dec_slf_attn, dec_enc_slf_attn, disentangled_weight
class DecoderLayer_Big(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.decoder_attention = TypeAreaMultiHeadAttention_Big(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_area = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_shot = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_bh = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_ba = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_bb = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_pan = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_oan = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_area = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_shot = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_bh = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_ba = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_bb = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_pan = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_oan = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self,decode_area, decode_shot, decode_bh, decode_ba, decode_bb, decode_pan, decode_oan,
                encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, slf_attn_mask=None, dec_enc_attn_mask=None, return_attns=None):
        decode_output, dec_slf_attn, disentangled_weight = self.decoder_attention(decode_area, decode_area, decode_area, 
                                                                                  decode_shot, decode_shot, decode_shot, 
                                                                                  decode_bh, decode_bh, decode_bh,
                                                                                  decode_ba, decode_ba, decode_ba,
                                                                                  decode_bb, decode_bb, decode_bb,
                                                                                  decode_pan, decode_pan, decode_pan,
                                                                                  decode_oan, decode_oan, decode_oan,
                                                                                  mask=slf_attn_mask, return_attns=return_attns)
        decode_output_area, dec_enc_slf_attn = self.decoder_encoder_attention_area(decode_output, encode_output_area, encode_output_area, mask=dec_enc_attn_mask)
        decode_output_shot, dec_enc_slf_attn = self.decoder_encoder_attention_shot(decode_output, encode_output_shot, encode_output_shot, mask=dec_enc_attn_mask)
        decode_output_bh, dec_enc_slf_attn = self.decoder_encoder_attention_bh(decode_output, encode_output_bh, encode_output_bh, mask=dec_enc_attn_mask)
        decode_output_ba, dec_enc_slf_attn = self.decoder_encoder_attention_ba(decode_output, encode_output_ba, encode_output_ba, mask=dec_enc_attn_mask)
        decode_output_bb, dec_enc_slf_attn = self.decoder_encoder_attention_bb(decode_output, encode_output_bb, encode_output_bb, mask=dec_enc_attn_mask)
        decode_output_pan, dec_enc_slf_attn = self.decoder_encoder_attention_pan(decode_output, encode_output_pan, encode_output_pan, mask=dec_enc_attn_mask)
        decode_output_oan, dec_enc_slf_attn = self.decoder_encoder_attention_oan(decode_output, encode_output_oan, encode_output_oan, mask=dec_enc_attn_mask)

        decode_output_area = self.pos_ffn_area(decode_output_area)
        decode_output_shot = self.pos_ffn_shot(decode_output_shot)
        decode_output_bh = self.pos_ffn_bh(decode_output_bh)
        decode_output_ba = self.pos_ffn_ba(decode_output_ba)
        decode_output_bb = self.pos_ffn_bb(decode_output_bb)
        decode_output_pan = self.pos_ffn_pan(decode_output_pan)
        decode_output_oan = self.pos_ffn_oan(decode_output_oan)
        return decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, dec_slf_attn, dec_enc_slf_attn, disentangled_weight
class DecoderLayer_Big2(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.decoder_attention = TypeAreaMultiHeadAttention_Big2(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_area = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_shot = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_bh = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_ba = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_bb = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_pan = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_oan = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_pl = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention_ol = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_area = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_shot = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_bh = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_ba = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_bb = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_pan = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_oan = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_pl = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn_ol = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self,decode_area, decode_shot, decode_bh, decode_ba, decode_bb, decode_pan, decode_oan, decode_pl, decode_ol,
                encode_output_area, encode_output_shot, encode_output_bh, encode_output_ba, encode_output_bb, encode_output_pan, encode_output_oan, encode_output_pl, encode_output_ol, slf_attn_mask=None, dec_enc_attn_mask=None, return_attns=None):
        decode_output, dec_slf_attn, disentangled_weight = self.decoder_attention(decode_area, decode_area, decode_area, 
                                                                                  decode_shot, decode_shot, decode_shot, 
                                                                                  decode_bh, decode_bh, decode_bh,
                                                                                  decode_ba, decode_ba, decode_ba,
                                                                                  decode_bb, decode_bb, decode_bb,
                                                                                  decode_pan, decode_pan, decode_pan,
                                                                                  decode_oan, decode_oan, decode_oan,
                                                                                  decode_pl, decode_pl, decode_pl,
                                                                                  decode_ol, decode_ol, decode_ol,
                                                                                  mask=slf_attn_mask, return_attns=return_attns)
        decode_output_area, dec_enc_slf_attn = self.decoder_encoder_attention_area(decode_output, encode_output_area, encode_output_area, mask=dec_enc_attn_mask)
        decode_output_shot, dec_enc_slf_attn = self.decoder_encoder_attention_shot(decode_output, encode_output_shot, encode_output_shot, mask=dec_enc_attn_mask)
        decode_output_bh, dec_enc_slf_attn = self.decoder_encoder_attention_bh(decode_output, encode_output_bh, encode_output_bh, mask=dec_enc_attn_mask)
        decode_output_ba, dec_enc_slf_attn = self.decoder_encoder_attention_ba(decode_output, encode_output_ba, encode_output_ba, mask=dec_enc_attn_mask)
        decode_output_bb, dec_enc_slf_attn = self.decoder_encoder_attention_bb(decode_output, encode_output_bb, encode_output_bb, mask=dec_enc_attn_mask)
        decode_output_pan, dec_enc_slf_attn = self.decoder_encoder_attention_pan(decode_output, encode_output_pan, encode_output_pan, mask=dec_enc_attn_mask)
        decode_output_oan, dec_enc_slf_attn = self.decoder_encoder_attention_oan(decode_output, encode_output_oan, encode_output_oan, mask=dec_enc_attn_mask)
        decode_output_pl, dec_enc_slf_attn = self.decoder_encoder_attention_pl(decode_output, encode_output_pl, encode_output_pl, mask=dec_enc_attn_mask)
        decode_output_ol, dec_enc_slf_attn = self.decoder_encoder_attention_ol(decode_output, encode_output_ol, encode_output_ol, mask=dec_enc_attn_mask)

        decode_output_area = self.pos_ffn_area(decode_output_area)
        decode_output_shot = self.pos_ffn_shot(decode_output_shot)
        decode_output_bh = self.pos_ffn_bh(decode_output_bh)
        decode_output_ba = self.pos_ffn_ba(decode_output_ba)
        decode_output_bb = self.pos_ffn_bb(decode_output_bb)
        decode_output_pan = self.pos_ffn_pan(decode_output_pan)
        decode_output_oan = self.pos_ffn_oan(decode_output_oan)
        decode_output_pl = self.pos_ffn_pl(decode_output_pl)
        decode_output_ol = self.pos_ffn_ol(decode_output_ol)
        return decode_output_area, decode_output_shot, decode_output_bh, decode_output_ba, decode_output_bb, decode_output_pan, decode_output_oan, decode_output_pl, decode_output_ol, dec_slf_attn, dec_enc_slf_attn, disentangled_weight


class GatedFusionLayer(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden1 = nn.Linear(d, d_inner, bias=False)
        self.hidden2 = nn.Linear(d, d_inner, bias=False)
        self.hidden3 = nn.Linear(d, d_inner, bias=False)
        self.gated1 = nn.Linear(d_inner*n, d, bias=False)
        self.gated2 = nn.Linear(d_inner*n, d, bias=False)
        self.gated3 = nn.Linear(d_inner*n, d, bias=False)

        self.decode_length = n_position - encode_length

        self.w_A = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A, x_B, x_L):
        # (batch, seq, dim)
        batch, seq_len, dim = x_A.shape
        w_A = self.w_A.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B = self.w_B.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L = self.w_L.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A = self.tanh_f(self.hidden1(x_A))
        h_B = self.tanh_f(self.hidden2(x_B))
        h_L = self.tanh_f(self.hidden3(x_L))

        x = torch.cat((x_A, x_B, x_L), dim=-1)
        z1 = self.sigmoid_f(self.gated1(x)) * h_A
        z2 = self.sigmoid_f(self.gated2(x)) * h_B
        z3 = self.sigmoid_f(self.gated3(x)) * h_L

        z1 = w_A[:, :seq_len, :] * z1
        z2 = w_B[:, :seq_len, :] * z2
        z3 = w_L[:, :seq_len, :] * z3

        return self.sigmoid_f(z1 + z2 + z3)
class GatedFusionLayer2(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden1 = nn.Linear(d, d_inner, bias=False)
        self.hidden2 = nn.Linear(d, d_inner, bias=False)
        self.hidden3 = nn.Linear(d, d_inner, bias=False)
        self.hidden4 = nn.Linear(d, d_inner, bias=False)
        self.hidden5 = nn.Linear(d, d_inner, bias=False)
        self.hidden6 = nn.Linear(d, d_inner, bias=False)
        self.gated1 = nn.Linear(d_inner*n, d, bias=False)
        self.gated2 = nn.Linear(d_inner*n, d, bias=False)
        self.gated3 = nn.Linear(d_inner*n, d, bias=False)
        self.gated4 = nn.Linear(d_inner*n, d, bias=False)
        self.gated5 = nn.Linear(d_inner*n, d, bias=False)
        self.gated6 = nn.Linear(d_inner*n, d, bias=False)

        self.decode_length = n_position - encode_length

        self.w_A_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.w_A_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A_area, x_B_area, x_L_area, x_A_shot, x_B_shot, x_L_shot):
        # (batch, seq, dim)
        batch, seq_len, dim = x_A_area.shape
        w_A_area = self.w_A_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_area = self.w_B_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_area = self.w_L_area.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_area = self.tanh_f(self.hidden1(x_A_area))
        h_B_area = self.tanh_f(self.hidden2(x_B_area))
        h_L_area = self.tanh_f(self.hidden3(x_L_area))

        x_area = torch.cat((x_A_area, x_B_area, x_L_area), dim=-1)
        z1_area = self.sigmoid_f(self.gated1(x_area)) * h_A_area
        z2_area = self.sigmoid_f(self.gated2(x_area)) * h_B_area
        z3_area = self.sigmoid_f(self.gated3(x_area)) * h_L_area

        z1_area = w_A_area[:, :seq_len, :] * z1_area
        z2_area = w_B_area[:, :seq_len, :] * z2_area
        z3_area = w_L_area[:, :seq_len, :] * z3_area


        batch, seq_len, dim = x_A_shot.shape
        w_A_shot = self.w_A_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_shot = self.w_B_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_shot = self.w_L_shot.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_shot = self.tanh_f(self.hidden1(x_A_shot))
        h_B_shot = self.tanh_f(self.hidden2(x_B_shot))
        h_L_shot = self.tanh_f(self.hidden3(x_L_shot))

        x_shot = torch.cat((x_A_shot, x_B_shot, x_L_shot), dim=-1)
        z1_shot = self.sigmoid_f(self.gated1(x_shot)) * h_A_shot
        z2_shot = self.sigmoid_f(self.gated2(x_shot)) * h_B_shot
        z3_shot = self.sigmoid_f(self.gated3(x_shot)) * h_L_shot

        z1_shot = w_A_shot[:, :seq_len, :] * z1_shot
        z2_shot = w_B_shot[:, :seq_len, :] * z2_shot
        z3_shot = w_L_shot[:, :seq_len, :] * z3_shot

        return self.sigmoid_f(z1_area + z2_area + z3_area + z1_shot + z2_shot + z3_shot)
class GatedFusionLayer3(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden1 = nn.Linear(d, d_inner, bias=False)
        self.hidden2 = nn.Linear(d, d_inner, bias=False)
        self.hidden3 = nn.Linear(d, d_inner, bias=False)
        self.hidden4 = nn.Linear(d, d_inner, bias=False)
        self.hidden5 = nn.Linear(d, d_inner, bias=False)
        self.hidden6 = nn.Linear(d, d_inner, bias=False)
        self.gated1 = nn.Linear(d_inner*n, d, bias=False)
        self.gated2 = nn.Linear(d_inner*n, d, bias=False)
        self.gated3 = nn.Linear(d_inner*n, d, bias=False)
        self.gated4 = nn.Linear(d_inner*n, d, bias=False)
        self.gated5 = nn.Linear(d_inner*n, d, bias=False)
        self.gated6 = nn.Linear(d_inner*n, d, bias=False)

        self.decode_length = n_position - encode_length

        self.w_A_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.w_A_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A_area, x_B_area, x_L_area, x_A_shot, x_B_shot, x_L_shot):
        # (batch, seq, dim)
        batch, seq_len, dim = x_A_area.shape
        w_A_area = self.w_A_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_area = self.w_B_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_area = self.w_L_area.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_area = self.tanh_f(self.hidden1(x_A_area))
        h_B_area = self.tanh_f(self.hidden2(x_B_area))
        h_L_area = self.tanh_f(self.hidden3(x_L_area))

        x_area = torch.cat((x_A_area, x_B_area, x_L_area), dim=-1)
        z1_area = self.sigmoid_f(self.gated1(x_area)) * h_A_area
        z2_area = self.sigmoid_f(self.gated2(x_area)) * h_B_area
        z3_area = self.sigmoid_f(self.gated3(x_area)) * h_L_area

        z1_area = w_A_area[:, :seq_len, :] * z1_area
        z2_area = w_B_area[:, :seq_len, :] * z2_area
        z3_area = w_L_area[:, :seq_len, :] * z3_area


        batch, seq_len, dim = x_A_shot.shape
        w_A_shot = self.w_A_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_shot = self.w_B_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_shot = self.w_L_shot.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_shot = self.tanh_f(self.hidden4(x_A_shot))
        h_B_shot = self.tanh_f(self.hidden5(x_B_shot))
        h_L_shot = self.tanh_f(self.hidden6(x_L_shot))

        x_shot = torch.cat((x_A_shot, x_B_shot, x_L_shot), dim=-1)
        z1_shot = self.sigmoid_f(self.gated4(x_shot)) * h_A_shot
        z2_shot = self.sigmoid_f(self.gated5(x_shot)) * h_B_shot
        z3_shot = self.sigmoid_f(self.gated6(x_shot)) * h_L_shot

        z1_shot = w_A_shot[:, :seq_len, :] * z1_shot
        z2_shot = w_B_shot[:, :seq_len, :] * z2_shot
        z3_shot = w_L_shot[:, :seq_len, :] * z3_shot

        return self.sigmoid_f(z1_area + z2_area + z3_area), self.sigmoid_f(z1_shot + z2_shot + z3_shot)
class GatedFusionLayer_model3(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden1 = nn.Linear(d, d_inner, bias=False)
        self.hidden2 = nn.Linear(d, d_inner, bias=False)
        self.hidden3 = nn.Linear(d, d_inner, bias=False)
        self.hidden4 = nn.Linear(d, d_inner, bias=False)
        self.hidden5 = nn.Linear(d, d_inner, bias=False)
        self.hidden6 = nn.Linear(d, d_inner, bias=False)
        self.gated1 = nn.Linear(d_inner*n, d, bias=False)
        self.gated2 = nn.Linear(d_inner*n, d, bias=False)
        self.gated3 = nn.Linear(d_inner*n, d, bias=False)
        self.gated4 = nn.Linear(d_inner*n, d, bias=False)
        self.gated5 = nn.Linear(d_inner*n, d, bias=False)
        self.gated6 = nn.Linear(d_inner*n, d, bias=False)

        self.decode_length = n_position - encode_length

        self.w_A_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.w_A_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A_area, x_B_area, x_L_area, x_A_shot, x_B_shot, x_L_shot):
        # (batch, seq, dim)
        batch, seq_len, dim = x_A_area.shape
        w_A_area = self.w_A_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_area = self.w_B_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_area = self.w_L_area.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_area = self.tanh_f(self.hidden1(x_A_area))
        h_B_area = self.tanh_f(self.hidden2(x_B_area))
        h_L_area = self.tanh_f(self.hidden3(x_L_area))

        x_area = torch.cat((x_A_area, x_B_area, x_L_area), dim=-1)
        z1_area = self.sigmoid_f(self.gated1(x_area)) * h_A_area
        z2_area = self.sigmoid_f(self.gated2(x_area)) * h_B_area
        z3_area = self.sigmoid_f(self.gated3(x_area)) * h_L_area

        z1_area = w_A_area[:, :seq_len, :] * z1_area
        z2_area = w_B_area[:, :seq_len, :] * z2_area
        z3_area = w_L_area[:, :seq_len, :] * z3_area


        batch, seq_len, dim = x_A_shot.shape
        w_A_shot = self.w_A_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_shot = self.w_B_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_shot = self.w_L_shot.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_shot = self.tanh_f(self.hidden4(x_A_shot))
        h_B_shot = self.tanh_f(self.hidden5(x_B_shot))
        h_L_shot = self.tanh_f(self.hidden6(x_L_shot))

        x_shot = torch.cat((x_A_shot, x_B_shot, x_L_shot), dim=-1)
        z1_shot = self.sigmoid_f(self.gated4(x_shot)) * h_A_shot
        z2_shot = self.sigmoid_f(self.gated5(x_shot)) * h_B_shot
        z3_shot = self.sigmoid_f(self.gated6(x_shot)) * h_L_shot

        z1_shot = w_A_shot[:, :seq_len, :] * z1_shot
        z2_shot = w_B_shot[:, :seq_len, :] * z2_shot
        z3_shot = w_L_shot[:, :seq_len, :] * z3_shot

        return self.sigmoid_f(z1_area + z2_area + z3_area), self.sigmoid_f(z1_shot + z2_shot + z3_shot)
class GatedFusionLayer_model3_ver2(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden1 = nn.Linear(d, d_inner, bias=False)
        self.hidden2 = nn.Linear(d, d_inner, bias=False)
        self.hidden3 = nn.Linear(d, d_inner, bias=False)
        self.hidden4 = nn.Linear(d, d_inner, bias=False)
        self.hidden5 = nn.Linear(d, d_inner, bias=False)
        self.hidden6 = nn.Linear(d, d_inner, bias=False)
        self.gated1 = nn.Linear(d_inner*n, d, bias=False)
        self.gated2 = nn.Linear(d_inner*n, d, bias=False)
        self.gated3 = nn.Linear(d_inner*n, d, bias=False)
        self.gated4 = nn.Linear(d_inner*n, d, bias=False)
        self.gated5 = nn.Linear(d_inner*n, d, bias=False)
        self.gated6 = nn.Linear(d_inner*n, d, bias=False)

        self.decode_length = n_position - encode_length

        self.w_A_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.w_A_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A_area, x_B_area, x_L_area, x_A_shot, x_B_shot, x_L_shot):
        # (batch, seq, dim)
        batch, seq_len, dim = x_A_area.shape
        w_A_area = self.w_A_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_area = self.w_B_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_area = self.w_L_area.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_area = self.tanh_f(self.hidden1(x_A_area))
        h_B_area = self.tanh_f(self.hidden2(x_B_area))
        h_L_area = self.tanh_f(self.hidden3(x_L_area))

        x_area = torch.cat((x_A_area, x_B_area, x_L_area), dim=-1)
        z1_area = self.sigmoid_f(self.gated1(x_area)) * h_A_area
        z2_area = self.sigmoid_f(self.gated2(x_area)) * h_B_area
        z3_area = self.sigmoid_f(self.gated3(x_area)) * h_L_area

        z1_area = w_A_area[:, :seq_len, :] * z1_area
        z2_area = w_B_area[:, :seq_len, :] * z2_area
        z3_area = w_L_area[:, :seq_len, :] * z3_area


        batch, seq_len, dim = x_A_shot.shape
        w_A_shot = self.w_A_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_shot = self.w_B_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_shot = self.w_L_shot.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_shot = self.tanh_f(self.hidden1(x_A_shot))
        h_B_shot = self.tanh_f(self.hidden2(x_B_shot))
        h_L_shot = self.tanh_f(self.hidden3(x_L_shot))

        x_shot = torch.cat((x_A_shot, x_B_shot, x_L_shot), dim=-1)
        z1_shot = self.sigmoid_f(self.gated1(x_shot)) * h_A_shot
        z2_shot = self.sigmoid_f(self.gated2(x_shot)) * h_B_shot
        z3_shot = self.sigmoid_f(self.gated3(x_shot)) * h_L_shot

        z1_shot = w_A_shot[:, :seq_len, :] * z1_shot
        z2_shot = w_B_shot[:, :seq_len, :] * z2_shot
        z3_shot = w_L_shot[:, :seq_len, :] * z3_shot

        return self.sigmoid_f(z1_area + z2_area + z3_area), self.sigmoid_f(z1_shot + z2_shot + z3_shot)
class GatedFusionLayer_model3_ver3(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden1 = nn.Linear(d, d_inner, bias=False)
        self.hidden2 = nn.Linear(d, d_inner, bias=False)
        self.hidden3 = nn.Linear(d, d_inner, bias=False)
        self.hidden4 = nn.Linear(d, d_inner, bias=False)
        self.hidden5 = nn.Linear(d, d_inner, bias=False)
        self.hidden6 = nn.Linear(d, d_inner, bias=False)
        self.gated1 = nn.Linear(d_inner*n, d, bias=False)
        self.gated2 = nn.Linear(d_inner*n, d, bias=False)
        self.gated3 = nn.Linear(d_inner*n, d, bias=False)
        self.gated4 = nn.Linear(d_inner*n, d, bias=False)
        self.gated5 = nn.Linear(d_inner*n, d, bias=False)
        self.gated6 = nn.Linear(d_inner*n, d, bias=False)

        self.decode_length = n_position - encode_length

        self.w_A_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.w_A_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A_area, x_B_area, x_L_area, x_A_shot, x_B_shot, x_L_shot):
        # (batch, seq, dim)
        batch, seq_len, dim = x_A_area.shape
        w_A_area = self.w_A_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_area = self.w_B_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_area = self.w_L_area.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_area = self.tanh_f(self.hidden1(x_A_area))
        h_B_area = self.tanh_f(self.hidden2(x_B_area))
        h_L_area = self.tanh_f(self.hidden3(x_L_area))

        x_area = torch.cat((x_A_area, x_B_area, x_L_area), dim=-1)
        z1_area = self.sigmoid_f(self.gated1(x_area)) * h_A_area
        z2_area = self.sigmoid_f(self.gated2(x_area)) * h_B_area
        z3_area = self.sigmoid_f(self.gated3(x_area)) * h_L_area

        z1_area = w_A_area[:, :seq_len, :] * z1_area
        z2_area = w_B_area[:, :seq_len, :] * z2_area
        z3_area = w_L_area[:, :seq_len, :] * z3_area


        batch, seq_len, dim = x_A_shot.shape
        w_A_shot = self.w_A_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_shot = self.w_B_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_shot = self.w_L_shot.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_shot = self.tanh_f(self.hidden1(x_A_shot))
        h_B_shot = self.tanh_f(self.hidden2(x_B_shot))
        h_L_shot = self.tanh_f(self.hidden3(x_L_shot))

        x_shot = torch.cat((x_A_shot, x_B_shot, x_L_shot), dim=-1)
        z1_shot = self.sigmoid_f(self.gated1(x_shot)) * h_A_shot
        z2_shot = self.sigmoid_f(self.gated2(x_shot)) * h_B_shot
        z3_shot = self.sigmoid_f(self.gated3(x_shot)) * h_L_shot

        z1_shot = w_A_shot[:, :seq_len, :] * z1_shot
        z2_shot = w_B_shot[:, :seq_len, :] * z2_shot
        z3_shot = w_L_shot[:, :seq_len, :] * z3_shot

        return self.sigmoid_f(z1_area + z2_area + z3_area + z1_shot + z2_shot + z3_shot)
class GatedFusionLayer_Big(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden_x_A_area = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_area = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_area = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_shot = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_shot = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_shot = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_bh = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_bh = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_bh = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_ba = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_ba = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_ba = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_bb = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_bb = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_bb = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_pan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_pan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_pan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_oan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_oan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_oan = nn.Linear(d, d_inner, bias=False)

        self.gated_x_A_area = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_area = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_area = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_shot = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_shot = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_shot = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_bh = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_bh = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_bh = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_ba = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_ba = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_ba = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_bb = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_bb = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_bb = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_pan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_pan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_pan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_oan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_oan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_oan = nn.Linear(d_inner*n, d, bias=False)

        self.decode_length = n_position - encode_length

        self.w_A_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_bh = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_bh = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_bh = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_ba = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_ba = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_ba = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_bb = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_bb = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_bb = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_pan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_pan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_pan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_oan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_oan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_oan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A_area, x_B_area, x_L_area, x_A_shot, x_B_shot, x_L_shot,
                x_A_bh, x_B_bh, x_L_bh, x_A_ba, x_B_ba, x_L_ba,
                x_A_bb, x_B_bb, x_L_bb, x_A_pan, x_B_pan, x_L_pan, x_A_oan, x_B_oan, x_L_oan):
        # (batch, seq, dim)
        batch, seq_len, dim = x_A_area.shape
        w_A_area = self.w_A_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_area = self.w_B_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_area = self.w_L_area.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_area = self.tanh_f(self.hidden_x_A_area(x_A_area))
        h_B_area = self.tanh_f(self.hidden_x_B_area(x_B_area))
        h_L_area = self.tanh_f(self.hidden_x_L_area(x_L_area))

        x_area = torch.cat((x_A_area, x_B_area, x_L_area), dim=-1)
        z1_area = self.sigmoid_f(self.gated_x_A_area(x_area)) * h_A_area
        z2_area = self.sigmoid_f(self.gated_x_B_area(x_area)) * h_B_area
        z3_area = self.sigmoid_f(self.gated_x_L_area(x_area)) * h_L_area

        z1_area = w_A_area[:, :seq_len, :] * z1_area
        z2_area = w_B_area[:, :seq_len, :] * z2_area
        z3_area = w_L_area[:, :seq_len, :] * z3_area

        ####################
        batch, seq_len, dim = x_A_shot.shape
        w_A_shot = self.w_A_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_shot = self.w_B_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_shot = self.w_L_shot.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_shot = self.tanh_f(self.hidden_x_A_shot(x_A_shot))
        h_B_shot = self.tanh_f(self.hidden_x_B_shot(x_B_shot))
        h_L_shot = self.tanh_f(self.hidden_x_L_shot(x_L_shot))

        x_shot = torch.cat((x_A_shot, x_B_shot, x_L_shot), dim=-1)
        z1_shot = self.sigmoid_f(self.gated_x_A_shot(x_shot)) * h_A_shot
        z2_shot = self.sigmoid_f(self.gated_x_B_shot(x_shot)) * h_B_shot
        z3_shot = self.sigmoid_f(self.gated_x_L_shot(x_shot)) * h_L_shot

        z1_shot = w_A_shot[:, :seq_len, :] * z1_shot
        z2_shot = w_B_shot[:, :seq_len, :] * z2_shot
        z3_shot = w_L_shot[:, :seq_len, :] * z3_shot

        ####################
        batch, seq_len, dim = x_A_bh.shape
        w_A_bh = self.w_A_bh.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_bh = self.w_B_bh.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_bh = self.w_L_bh.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_bh = self.tanh_f(self.hidden_x_A_bh(x_A_bh))
        h_B_bh = self.tanh_f(self.hidden_x_B_bh(x_B_bh))
        h_L_bh = self.tanh_f(self.hidden_x_L_bh(x_L_bh))

        x_bh = torch.cat((x_A_bh, x_B_bh, x_L_bh), dim=-1)
        z1_bh = self.sigmoid_f(self.gated_x_A_bh(x_bh)) * h_A_bh
        z2_bh = self.sigmoid_f(self.gated_x_B_bh(x_bh)) * h_B_bh
        z3_bh = self.sigmoid_f(self.gated_x_L_bh(x_bh)) * h_L_bh

        z1_bh = w_A_bh[:, :seq_len, :] * z1_bh
        z2_bh = w_B_bh[:, :seq_len, :] * z2_bh
        z3_bh = w_L_bh[:, :seq_len, :] * z3_bh

        ####################
        batch, seq_len, dim = x_A_ba.shape
        w_A_ba = self.w_A_ba.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_ba = self.w_B_ba.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_ba = self.w_L_ba.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_ba = self.tanh_f(self.hidden_x_A_ba(x_A_ba))
        h_B_ba = self.tanh_f(self.hidden_x_B_ba(x_B_ba))
        h_L_ba = self.tanh_f(self.hidden_x_L_ba(x_L_ba))

        x_ba = torch.cat((x_A_ba, x_B_ba, x_L_ba), dim=-1)
        z1_ba = self.sigmoid_f(self.gated_x_A_ba(x_ba)) * h_A_ba
        z2_ba = self.sigmoid_f(self.gated_x_B_ba(x_ba)) * h_B_ba
        z3_ba = self.sigmoid_f(self.gated_x_L_ba(x_ba)) * h_L_ba

        z1_ba = w_A_ba[:, :seq_len, :] * z1_ba
        z2_ba = w_B_ba[:, :seq_len, :] * z2_ba
        z3_ba = w_L_ba[:, :seq_len, :] * z3_ba

        ####################
        batch, seq_len, dim = x_A_bb.shape
        w_A_bb = self.w_A_bb.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_bb = self.w_B_bb.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_bb = self.w_L_bb.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_bb = self.tanh_f(self.hidden_x_A_bb(x_A_bb))
        h_B_bb = self.tanh_f(self.hidden_x_B_bb(x_B_bb))
        h_L_bb = self.tanh_f(self.hidden_x_L_bb(x_L_bb))

        x_bb = torch.cat((x_A_bb, x_B_bb, x_L_bb), dim=-1)
        z1_bb = self.sigmoid_f(self.gated_x_A_bb(x_bb)) * h_A_bb
        z2_bb = self.sigmoid_f(self.gated_x_B_bb(x_bb)) * h_B_bb
        z3_bb = self.sigmoid_f(self.gated_x_L_bb(x_bb)) * h_L_bb

        z1_bb = w_A_bb[:, :seq_len, :] * z1_bb
        z2_bb = w_B_bb[:, :seq_len, :] * z2_bb
        z3_bb = w_L_bb[:, :seq_len, :] * z3_bb

        ####################
        batch, seq_len, dim = x_A_pan.shape
        w_A_pan = self.w_A_pan.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_pan = self.w_B_pan.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_pan = self.w_L_pan.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_pan = self.tanh_f(self.hidden_x_A_pan(x_A_pan))
        h_B_pan = self.tanh_f(self.hidden_x_B_pan(x_B_pan))
        h_L_pan = self.tanh_f(self.hidden_x_L_pan(x_L_pan))

        x_pan = torch.cat((x_A_pan, x_B_pan, x_L_pan), dim=-1)
        z1_pan = self.sigmoid_f(self.gated_x_A_pan(x_pan)) * h_A_pan
        z2_pan = self.sigmoid_f(self.gated_x_B_pan(x_pan)) * h_B_pan
        z3_pan = self.sigmoid_f(self.gated_x_L_pan(x_pan)) * h_L_pan

        z1_pan = w_A_pan[:, :seq_len, :] * z1_pan
        z2_pan = w_B_pan[:, :seq_len, :] * z2_pan
        z3_pan = w_L_pan[:, :seq_len, :] * z3_pan

        ####################
        batch, seq_len, dim = x_A_oan.shape
        w_A_oan = self.w_A_oan.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_oan = self.w_B_oan.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_oan = self.w_L_oan.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_oan = self.tanh_f(self.hidden_x_A_oan(x_A_oan))
        h_B_oan = self.tanh_f(self.hidden_x_B_oan(x_B_oan))
        h_L_oan = self.tanh_f(self.hidden_x_L_oan(x_L_oan))

        x_oan = torch.cat((x_A_oan, x_B_oan, x_L_oan), dim=-1)
        z1_oan = self.sigmoid_f(self.gated_x_A_oan(x_oan)) * h_A_oan
        z2_oan = self.sigmoid_f(self.gated_x_B_oan(x_oan)) * h_B_oan
        z3_oan = self.sigmoid_f(self.gated_x_L_oan(x_oan)) * h_L_oan

        z1_oan = w_A_oan[:, :seq_len, :] * z1_oan
        z2_oan = w_B_oan[:, :seq_len, :] * z2_oan
        z3_oan = w_L_oan[:, :seq_len, :] * z3_oan

        return self.sigmoid_f(z1_area + z2_area + z3_area), self.sigmoid_f(z1_shot + z2_shot + z3_shot), self.sigmoid_f(z1_bh + z2_bh + z3_bh), self.sigmoid_f(z1_ba + z2_ba + z3_ba), self.sigmoid_f(z1_bb + z2_bb + z3_bb), self.sigmoid_f(z1_pan + z2_pan + z3_pan), self.sigmoid_f(z1_oan + z2_oan + z3_oan)
class GatedFusionLayer_Big2(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden_x_A_area = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_area = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_area = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_shot = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_shot = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_shot = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_bh = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_bh = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_bh = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_ba = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_ba = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_ba = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_bb = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_bb = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_bb = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_pan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_pan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_pan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_oan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_oan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_oan = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_pl = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_pl = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_pl = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_A_ol = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_B_ol = nn.Linear(d, d_inner, bias=False)
        self.hidden_x_L_ol = nn.Linear(d, d_inner, bias=False)

        self.gated_x_A_area = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_area = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_area = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_shot = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_shot = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_shot = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_bh = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_bh = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_bh = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_ba = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_ba = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_ba = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_bb = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_bb = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_bb = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_pan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_pan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_pan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_oan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_oan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_oan = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_pl = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_pl = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_pl = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_A_ol = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_B_ol = nn.Linear(d_inner*n, d, bias=False)
        self.gated_x_L_ol = nn.Linear(d_inner*n, d, bias=False)

        self.decode_length = n_position - encode_length

        self.w_A_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_area = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_shot = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_bh = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_bh = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_bh = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_ba = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_ba = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_ba = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_bb = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_bb = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_bb = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_pan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_pan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_pan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_oan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_oan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_oan = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_pl = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_pl = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_pl = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_A_ol = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B_ol = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L_ol = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)

        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A_area, x_B_area, x_L_area, x_A_shot, x_B_shot, x_L_shot,
                x_A_bh, x_B_bh, x_L_bh, x_A_ba, x_B_ba, x_L_ba,
                x_A_bb, x_B_bb, x_L_bb, x_A_pan, x_B_pan, x_L_pan, 
                x_A_oan, x_B_oan, x_L_oan, x_A_pl, x_B_pl, x_L_pl, x_A_ol, x_B_ol, x_L_ol):
        # (batch, seq, dim)
        batch, seq_len, dim = x_A_area.shape
        w_A_area = self.w_A_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_area = self.w_B_area.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_area = self.w_L_area.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_area = self.tanh_f(self.hidden_x_A_area(x_A_area))
        h_B_area = self.tanh_f(self.hidden_x_B_area(x_B_area))
        h_L_area = self.tanh_f(self.hidden_x_L_area(x_L_area))

        x_area = torch.cat((x_A_area, x_B_area, x_L_area), dim=-1)
        z1_area = self.sigmoid_f(self.gated_x_A_area(x_area)) * h_A_area
        z2_area = self.sigmoid_f(self.gated_x_B_area(x_area)) * h_B_area
        z3_area = self.sigmoid_f(self.gated_x_L_area(x_area)) * h_L_area

        z1_area = w_A_area[:, :seq_len, :] * z1_area
        z2_area = w_B_area[:, :seq_len, :] * z2_area
        z3_area = w_L_area[:, :seq_len, :] * z3_area

        ####################
        batch, seq_len, dim = x_A_shot.shape
        w_A_shot = self.w_A_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_shot = self.w_B_shot.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_shot = self.w_L_shot.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_shot = self.tanh_f(self.hidden_x_A_shot(x_A_shot))
        h_B_shot = self.tanh_f(self.hidden_x_B_shot(x_B_shot))
        h_L_shot = self.tanh_f(self.hidden_x_L_shot(x_L_shot))

        x_shot = torch.cat((x_A_shot, x_B_shot, x_L_shot), dim=-1)
        z1_shot = self.sigmoid_f(self.gated_x_A_shot(x_shot)) * h_A_shot
        z2_shot = self.sigmoid_f(self.gated_x_B_shot(x_shot)) * h_B_shot
        z3_shot = self.sigmoid_f(self.gated_x_L_shot(x_shot)) * h_L_shot

        z1_shot = w_A_shot[:, :seq_len, :] * z1_shot
        z2_shot = w_B_shot[:, :seq_len, :] * z2_shot
        z3_shot = w_L_shot[:, :seq_len, :] * z3_shot

        ####################
        batch, seq_len, dim = x_A_bh.shape
        w_A_bh = self.w_A_bh.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_bh = self.w_B_bh.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_bh = self.w_L_bh.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_bh = self.tanh_f(self.hidden_x_A_bh(x_A_bh))
        h_B_bh = self.tanh_f(self.hidden_x_B_bh(x_B_bh))
        h_L_bh = self.tanh_f(self.hidden_x_L_bh(x_L_bh))

        x_bh = torch.cat((x_A_bh, x_B_bh, x_L_bh), dim=-1)
        z1_bh = self.sigmoid_f(self.gated_x_A_bh(x_bh)) * h_A_bh
        z2_bh = self.sigmoid_f(self.gated_x_B_bh(x_bh)) * h_B_bh
        z3_bh = self.sigmoid_f(self.gated_x_L_bh(x_bh)) * h_L_bh

        z1_bh = w_A_bh[:, :seq_len, :] * z1_bh
        z2_bh = w_B_bh[:, :seq_len, :] * z2_bh
        z3_bh = w_L_bh[:, :seq_len, :] * z3_bh

        ####################
        batch, seq_len, dim = x_A_ba.shape
        w_A_ba = self.w_A_ba.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_ba = self.w_B_ba.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_ba = self.w_L_ba.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_ba = self.tanh_f(self.hidden_x_A_ba(x_A_ba))
        h_B_ba = self.tanh_f(self.hidden_x_B_ba(x_B_ba))
        h_L_ba = self.tanh_f(self.hidden_x_L_ba(x_L_ba))

        x_ba = torch.cat((x_A_ba, x_B_ba, x_L_ba), dim=-1)
        z1_ba = self.sigmoid_f(self.gated_x_A_ba(x_ba)) * h_A_ba
        z2_ba = self.sigmoid_f(self.gated_x_B_ba(x_ba)) * h_B_ba
        z3_ba = self.sigmoid_f(self.gated_x_L_ba(x_ba)) * h_L_ba

        z1_ba = w_A_ba[:, :seq_len, :] * z1_ba
        z2_ba = w_B_ba[:, :seq_len, :] * z2_ba
        z3_ba = w_L_ba[:, :seq_len, :] * z3_ba

        ####################
        batch, seq_len, dim = x_A_bb.shape
        w_A_bb = self.w_A_bb.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_bb = self.w_B_bb.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_bb = self.w_L_bb.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_bb = self.tanh_f(self.hidden_x_A_bb(x_A_bb))
        h_B_bb = self.tanh_f(self.hidden_x_B_bb(x_B_bb))
        h_L_bb = self.tanh_f(self.hidden_x_L_bb(x_L_bb))

        x_bb = torch.cat((x_A_bb, x_B_bb, x_L_bb), dim=-1)
        z1_bb = self.sigmoid_f(self.gated_x_A_bb(x_bb)) * h_A_bb
        z2_bb = self.sigmoid_f(self.gated_x_B_bb(x_bb)) * h_B_bb
        z3_bb = self.sigmoid_f(self.gated_x_L_bb(x_bb)) * h_L_bb

        z1_bb = w_A_bb[:, :seq_len, :] * z1_bb
        z2_bb = w_B_bb[:, :seq_len, :] * z2_bb
        z3_bb = w_L_bb[:, :seq_len, :] * z3_bb

        ####################
        batch, seq_len, dim = x_A_pan.shape
        w_A_pan = self.w_A_pan.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_pan = self.w_B_pan.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_pan = self.w_L_pan.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_pan = self.tanh_f(self.hidden_x_A_pan(x_A_pan))
        h_B_pan = self.tanh_f(self.hidden_x_B_pan(x_B_pan))
        h_L_pan = self.tanh_f(self.hidden_x_L_pan(x_L_pan))

        x_pan = torch.cat((x_A_pan, x_B_pan, x_L_pan), dim=-1)
        z1_pan = self.sigmoid_f(self.gated_x_A_pan(x_pan)) * h_A_pan
        z2_pan = self.sigmoid_f(self.gated_x_B_pan(x_pan)) * h_B_pan
        z3_pan = self.sigmoid_f(self.gated_x_L_pan(x_pan)) * h_L_pan

        z1_pan = w_A_pan[:, :seq_len, :] * z1_pan
        z2_pan = w_B_pan[:, :seq_len, :] * z2_pan
        z3_pan = w_L_pan[:, :seq_len, :] * z3_pan

        ####################
        batch, seq_len, dim = x_A_oan.shape
        w_A_oan = self.w_A_oan.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_oan = self.w_B_oan.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_oan = self.w_L_oan.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_oan = self.tanh_f(self.hidden_x_A_oan(x_A_oan))
        h_B_oan = self.tanh_f(self.hidden_x_B_oan(x_B_oan))
        h_L_oan = self.tanh_f(self.hidden_x_L_oan(x_L_oan))

        x_oan = torch.cat((x_A_oan, x_B_oan, x_L_oan), dim=-1)
        z1_oan = self.sigmoid_f(self.gated_x_A_oan(x_oan)) * h_A_oan
        z2_oan = self.sigmoid_f(self.gated_x_B_oan(x_oan)) * h_B_oan
        z3_oan = self.sigmoid_f(self.gated_x_L_oan(x_oan)) * h_L_oan

        z1_oan = w_A_oan[:, :seq_len, :] * z1_oan
        z2_oan = w_B_oan[:, :seq_len, :] * z2_oan
        z3_oan = w_L_oan[:, :seq_len, :] * z3_oan

        ####################
        batch, seq_len, dim = x_A_pl.shape
        w_A_pl = self.w_A_pl.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_pl = self.w_B_pl.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_pl = self.w_L_pl.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_pl = self.tanh_f(self.hidden_x_A_pl(x_A_pl))
        h_B_pl = self.tanh_f(self.hidden_x_B_pl(x_B_pl))
        h_L_pl = self.tanh_f(self.hidden_x_L_pl(x_L_pl))

        x_pl = torch.cat((x_A_pl, x_B_pl, x_L_pl), dim=-1)
        z1_pl = self.sigmoid_f(self.gated_x_A_pl(x_pl)) * h_A_pl
        z2_pl = self.sigmoid_f(self.gated_x_B_pl(x_pl)) * h_B_pl
        z3_pl = self.sigmoid_f(self.gated_x_L_pl(x_pl)) * h_L_pl

        z1_pl = w_A_pl[:, :seq_len, :] * z1_pl
        z2_pl = w_B_pl[:, :seq_len, :] * z2_pl
        z3_pl = w_L_pl[:, :seq_len, :] * z3_pl

        ####################
        batch, seq_len, dim = x_A_ol.shape
        w_A_ol = self.w_A_ol.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_B_ol = self.w_B_ol.unsqueeze(0).repeat_interleave(batch, dim=0)                  # (batch, seq_len, dim)
        w_L_ol = self.w_L_ol.unsqueeze(0).repeat_interleave(batch, dim=0) 

        h_A_ol = self.tanh_f(self.hidden_x_A_ol(x_A_ol))
        h_B_ol = self.tanh_f(self.hidden_x_B_ol(x_B_ol))
        h_L_ol = self.tanh_f(self.hidden_x_L_ol(x_L_ol))

        x_ol = torch.cat((x_A_ol, x_B_ol, x_L_ol), dim=-1)
        z1_ol = self.sigmoid_f(self.gated_x_A_ol(x_ol)) * h_A_ol
        z2_ol = self.sigmoid_f(self.gated_x_B_ol(x_ol)) * h_B_ol
        z3_ol = self.sigmoid_f(self.gated_x_L_ol(x_ol)) * h_L_ol

        z1_ol = w_A_ol[:, :seq_len, :] * z1_ol
        z2_ol = w_B_ol[:, :seq_len, :] * z2_ol
        z3_ol = w_L_ol[:, :seq_len, :] * z3_ol

        return self.sigmoid_f(z1_area + z2_area + z3_area), self.sigmoid_f(z1_shot + z2_shot + z3_shot), self.sigmoid_f(z1_bh + z2_bh + z3_bh), self.sigmoid_f(z1_ba + z2_ba + z3_ba), self.sigmoid_f(z1_bb + z2_bb + z3_bb), self.sigmoid_f(z1_pan + z2_pan + z3_pan), self.sigmoid_f(z1_oan + z2_oan + z3_oan), self.sigmoid_f(z1_pl + z2_pl + z3_pl), self.sigmoid_f(z1_ol + z2_ol + z3_ol)
        

    