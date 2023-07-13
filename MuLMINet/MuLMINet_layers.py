import torch.nn as nn
import torch
from MuLMINet.MuLMINet_submodules import TypeAreaMultiHeadAttention, MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer_MuLMINet(nn.Module):
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
class DecoderLayer_MuLMINet(nn.Module):
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

class GatedFusionLayer_MuLMINet_Variant1(nn.Module):
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
class GatedFusionLayer_MuLMINet_Variant2(nn.Module):
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

        

    