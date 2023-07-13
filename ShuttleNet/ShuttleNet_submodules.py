import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class MultiHeadAttention2(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q_area, k_area, v_area, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q_area.size(0), q_area.size(1), k_area.size(1), v_area.size(1)

        residual = q_area

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_area = self.w_qs(q_area).view(sz_b, len_q, n_head, d_k)
        k_area = self.w_ks(k_area).view(sz_b, len_k, n_head, d_k)
        v_area = self.w_vs(v_area).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q_area, k_area, v_area = q_area.transpose(1, 2), k_area.transpose(1, 2), v_area.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q_area, attn = self.attention(q_area, k_area, v_area, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q_area = q_area.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q_area = self.dropout(self.fc(q_area))
        q_area += residual

        q_area = self.layer_norm(q_area)

        return q_area, attn


class MultiHeadAttention2(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class TypeAreaScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention of type-area attention'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, mask=None, return_attns=None):
        a2a = torch.matmul(q_a, k_a.transpose(2, 3))
        a2s = torch.matmul(q_a, k_s.transpose(2, 3))
        s2a = torch.matmul(q_s, k_a.transpose(2, 3))
        s2s = torch.matmul(q_s, k_s.transpose(2, 3))
        attention_score = (a2a + a2s + s2a + s2s) / self.temperature

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        disentangled = {}
        if return_attns is not None:
            if mask is not None:
                disentangled['a2a'] = (a2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2s'] = (a2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2a'] = (s2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2s'] = (s2s / self.temperature).masked_fill(mask == 0, -1e9)
            else:
                disentangled['a2a'] = a2a / self.temperature
                disentangled['a2s'] = a2s / self.temperature
                disentangled['s2a'] = s2a / self.temperature
                disentangled['s2s'] = s2s / self.temperature

            disentangled['a2a'] = self.dropout(F.softmax(disentangled['a2a'], dim=-1))
            disentangled['a2s'] = self.dropout(F.softmax(disentangled['a2s'], dim=-1))
            disentangled['s2a'] = self.dropout(F.softmax(disentangled['s2a'], dim=-1))
            disentangled['s2s'] = self.dropout(F.softmax(disentangled['s2s'], dim=-1))

        attention_score = self.dropout(F.softmax(attention_score, dim=-1))
        output = torch.matmul(attention_score, (v_a + v_s))

        return output, attention_score, disentangled
class TypeAreaScaledDotProductAttention_Big(nn.Module):
    ''' Scaled Dot-Product Attention of type-area attention'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, q_bh, k_bh, v_bh, q_bah, k_bah, v_bah, q_bbh, k_bbh, v_bbh, q_pan, k_pan, v_pan, q_oan, k_oan, v_oan, mask=None, return_attns=None):
        a2a = torch.matmul(q_a, k_a.transpose(2, 3))
        a2s = torch.matmul(q_a, k_s.transpose(2, 3))
        a2bh = torch.matmul(q_a, k_bh.transpose(2, 3))
        a2bah = torch.matmul(q_a, k_bah.transpose(2, 3))
        a2bbh = torch.matmul(q_a, k_bbh.transpose(2, 3))
        a2pan = torch.matmul(q_a, k_pan.transpose(2, 3))
        a2oan = torch.matmul(q_a, k_oan.transpose(2, 3))

        s2a = torch.matmul(q_s, k_a.transpose(2, 3))
        s2s = torch.matmul(q_s, k_s.transpose(2, 3))
        s2bh = torch.matmul(q_s, k_bh.transpose(2, 3))
        s2bah = torch.matmul(q_s, k_bah.transpose(2, 3))
        s2bbh = torch.matmul(q_s, k_bbh.transpose(2, 3))
        s2pan = torch.matmul(q_s, k_pan.transpose(2, 3))
        s2oan = torch.matmul(q_s, k_oan.transpose(2, 3))

        bh2a = torch.matmul(q_bh, k_a.transpose(2, 3))
        bh2s = torch.matmul(q_bh, k_s.transpose(2, 3))
        bh2bh = torch.matmul(q_bh, k_bh.transpose(2, 3))
        bh2bah = torch.matmul(q_bh, k_bah.transpose(2, 3))
        bh2bbh = torch.matmul(q_bh, k_bbh.transpose(2, 3))
        bh2pan = torch.matmul(q_bh, k_pan.transpose(2, 3))
        bh2oan = torch.matmul(q_bh, k_oan.transpose(2, 3))

        bah2a = torch.matmul(q_bah, k_a.transpose(2, 3))
        bah2s = torch.matmul(q_bah, k_s.transpose(2, 3))
        bah2bh = torch.matmul(q_bah, k_bh.transpose(2, 3))
        bah2bah = torch.matmul(q_bah, k_bah.transpose(2, 3))
        bah2bbh = torch.matmul(q_bah, k_bbh.transpose(2, 3))
        bah2pan = torch.matmul(q_bah, k_pan.transpose(2, 3))
        bah2oan = torch.matmul(q_bah, k_oan.transpose(2, 3))

        bbh2a = torch.matmul(q_bbh, k_a.transpose(2, 3))
        bbh2s = torch.matmul(q_bbh, k_s.transpose(2, 3))
        bbh2bh = torch.matmul(q_bbh, k_bh.transpose(2, 3))
        bbh2bah = torch.matmul(q_bbh, k_bah.transpose(2, 3))
        bbh2bbh = torch.matmul(q_bbh, k_bbh.transpose(2, 3))
        bbh2pan = torch.matmul(q_bbh, k_pan.transpose(2, 3))
        bbh2oan = torch.matmul(q_bbh, k_oan.transpose(2, 3))

        pan2a = torch.matmul(q_pan, k_a.transpose(2, 3))
        pan2s = torch.matmul(q_pan, k_s.transpose(2, 3))
        pan2bh = torch.matmul(q_pan, k_bh.transpose(2, 3))
        pan2bah = torch.matmul(q_pan, k_bah.transpose(2, 3))
        pan2bbh = torch.matmul(q_pan, k_bbh.transpose(2, 3))
        pan2pan = torch.matmul(q_pan, k_pan.transpose(2, 3))
        pan2oan = torch.matmul(q_pan, k_oan.transpose(2, 3))

        oan2a = torch.matmul(q_oan, k_a.transpose(2, 3))
        oan2s = torch.matmul(q_oan, k_s.transpose(2, 3))
        oan2bh = torch.matmul(q_oan, k_bh.transpose(2, 3))
        oan2bah = torch.matmul(q_oan, k_bah.transpose(2, 3))
        oan2bbh = torch.matmul(q_oan, k_bbh.transpose(2, 3))
        oan2pan = torch.matmul(q_oan, k_pan.transpose(2, 3))
        oan2oan = torch.matmul(q_oan, k_oan.transpose(2, 3))

        attention_score = (a2a + a2s + a2bh + a2bah + a2bbh + a2pan + a2oan +
                           s2a + s2s + s2bh + s2bah + s2bbh + s2pan + s2oan +
                           bh2a + bh2s + bh2bh + bh2bah + bh2bbh + bh2pan + bh2oan +
                           bah2a + bah2s + bah2bh + bah2bah + bah2bbh + bah2pan + bah2oan +
                           bbh2a + bbh2s + bbh2bh + bbh2bah + bbh2bbh + bbh2pan + bbh2oan +
                           pan2a + pan2s + pan2bh + pan2bah + pan2bbh + pan2pan + pan2oan +
                           oan2a + oan2s + oan2bh + oan2bah + oan2bbh + oan2pan + oan2oan ) / self.temperature

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        disentangled = {}
        if return_attns is not None:
            if mask is not None:
                disentangled['a2a'] = (a2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2s'] = (a2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2bh'] = (a2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2bah'] = (a2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2bbh'] = (a2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2pan'] = (a2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2oan'] = (a2oan / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['s2a'] = (s2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2s'] = (s2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2bh'] = (s2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2bah'] = (s2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2bbh'] = (s2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2pan'] = (s2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2oan'] = (s2oan / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['bh2a'] = (bh2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2s'] = (bh2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2bh'] = (bh2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2bah'] = (bh2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2bbh'] = (bh2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2pan'] = (bh2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2oan'] = (bh2oan / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['bah2a'] = (bah2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2s'] = (bah2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2bh'] = (bah2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2bah'] = (bah2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2bbh'] = (bah2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2pan'] = (bah2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2oan'] = (bah2oan / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['bbh2a'] = (bbh2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2s'] = (bbh2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2bh'] = (bbh2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2bah'] = (bbh2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2bbh'] = (bbh2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2pan'] = (bbh2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2oan'] = (bbh2oan / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['pan2a'] = (pan2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2s'] = (pan2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2bh'] = (pan2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2bah'] = (pan2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2bbh'] = (pan2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2pan'] = (pan2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2oan'] = (pan2oan / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['oan2a'] = (oan2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2s'] = (oan2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2bh'] = (oan2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2bah'] = (oan2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2bbh'] = (oan2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2pan'] = (oan2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2oan'] = (oan2oan / self.temperature).masked_fill(mask == 0, -1e9)

            else:
                disentangled['a2a'] = (a2a / self.temperature)
                disentangled['a2s'] = (a2s / self.temperature)
                disentangled['a2bh'] = (a2bh / self.temperature)
                disentangled['a2bah'] = (a2bah / self.temperature)
                disentangled['a2bbh'] = (a2bbh / self.temperature)
                disentangled['a2pan'] = (a2pan / self.temperature)
                disentangled['a2oan'] = (a2oan / self.temperature)

                disentangled['s2a'] = (s2a / self.temperature)
                disentangled['s2s'] = (s2s / self.temperature)
                disentangled['s2bh'] = (s2bh / self.temperature)
                disentangled['s2bah'] = (s2bah / self.temperature)
                disentangled['s2bbh'] = (s2bbh / self.temperature)
                disentangled['s2pan'] = (s2pan / self.temperature)
                disentangled['s2oan'] = (s2oan / self.temperature)

                disentangled['bh2a'] = (bh2a / self.temperature)
                disentangled['bh2s'] = (bh2s / self.temperature)
                disentangled['bh2bh'] = (bh2bh / self.temperature)
                disentangled['bh2bah'] = (bh2bah / self.temperature)
                disentangled['bh2bbh'] = (bh2bbh / self.temperature)
                disentangled['bh2pan'] = (bh2pan / self.temperature)
                disentangled['bh2oan'] = (bh2oan / self.temperature)

                disentangled['bah2a'] = (bah2a / self.temperature)
                disentangled['bah2s'] = (bah2s / self.temperature)
                disentangled['bah2bh'] = (bah2bh / self.temperature)
                disentangled['bah2bah'] = (bah2bah / self.temperature)
                disentangled['bah2bbh'] = (bah2bbh / self.temperature)
                disentangled['bah2pan'] = (bah2pan / self.temperature)
                disentangled['bah2oan'] = (bah2oan / self.temperature)

                disentangled['bbh2a'] = (bbh2a / self.temperature)
                disentangled['bbh2s'] = (bbh2s / self.temperature)
                disentangled['bbh2bh'] = (bbh2bh / self.temperature)
                disentangled['bbh2bah'] = (bbh2bah / self.temperature)
                disentangled['bbh2bbh'] = (bbh2bbh / self.temperature)
                disentangled['bbh2pan'] = (bbh2pan / self.temperature)
                disentangled['bbh2oan'] = (bbh2oan / self.temperature)

                disentangled['pan2a'] = (pan2a / self.temperature)
                disentangled['pan2s'] = (pan2s / self.temperature)
                disentangled['pan2bh'] = (pan2bh / self.temperature)
                disentangled['pan2bah'] = (pan2bah / self.temperature)
                disentangled['pan2bbh'] = (pan2bbh / self.temperature)
                disentangled['pan2pan'] = (pan2pan / self.temperature)
                disentangled['pan2oan'] = (pan2oan / self.temperature)

                disentangled['oan2a'] = (oan2a / self.temperature)
                disentangled['oan2s'] = (oan2s / self.temperature)
                disentangled['oan2bh'] = (oan2bh / self.temperature)
                disentangled['oan2bah'] = (oan2bah / self.temperature)
                disentangled['oan2bbh'] = (oan2bbh / self.temperature)
                disentangled['oan2pan'] = (oan2pan / self.temperature)
                disentangled['oan2oan'] = (oan2oan / self.temperature)

            disentangled['a2a'] = self.dropout(F.softmax(disentangled['a2a'], dim=-1))
            disentangled['a2s'] = self.dropout(F.softmax(disentangled['a2s'], dim=-1))
            disentangled['a2bh'] = self.dropout(F.softmax(disentangled['a2bh'], dim=-1))
            disentangled['a2bah'] = self.dropout(F.softmax(disentangled['a2bah'], dim=-1))
            disentangled['a2bbh'] = self.dropout(F.softmax(disentangled['a2bbh'], dim=-1))
            disentangled['a2pan'] = self.dropout(F.softmax(disentangled['a2pan'], dim=-1))
            disentangled['a2oan'] = self.dropout(F.softmax(disentangled['a2oan'], dim=-1))

            disentangled['s2a'] = self.dropout(F.softmax(disentangled['s2a'], dim=-1))
            disentangled['s2s'] = self.dropout(F.softmax(disentangled['s2s'], dim=-1))
            disentangled['s2bh'] = self.dropout(F.softmax(disentangled['s2bh'], dim=-1))
            disentangled['s2bah'] = self.dropout(F.softmax(disentangled['s2bah'], dim=-1))
            disentangled['s2bbh'] = self.dropout(F.softmax(disentangled['s2bbh'], dim=-1))
            disentangled['s2pan'] = self.dropout(F.softmax(disentangled['s2pan'], dim=-1))
            disentangled['s2oan'] = self.dropout(F.softmax(disentangled['s2oan'], dim=-1))

            disentangled['bh2a'] = self.dropout(F.softmax(disentangled['bh2a'], dim=-1))
            disentangled['bh2s'] = self.dropout(F.softmax(disentangled['bh2s'], dim=-1))
            disentangled['bh2bh'] = self.dropout(F.softmax(disentangled['bh2bh'], dim=-1))
            disentangled['bh2bah'] = self.dropout(F.softmax(disentangled['bh2bah'], dim=-1))
            disentangled['bh2bbh'] = self.dropout(F.softmax(disentangled['bh2bbh'], dim=-1))
            disentangled['bh2pan'] = self.dropout(F.softmax(disentangled['bh2pan'], dim=-1))
            disentangled['bh2oan'] = self.dropout(F.softmax(disentangled['bh2oan'], dim=-1))

            disentangled['bah2a'] = self.dropout(F.softmax(disentangled['bah2a'], dim=-1))
            disentangled['bah2s'] = self.dropout(F.softmax(disentangled['bah2s'], dim=-1))
            disentangled['bah2bh'] = self.dropout(F.softmax(disentangled['bah2bh'], dim=-1))
            disentangled['bah2bah'] = self.dropout(F.softmax(disentangled['bah2bah'], dim=-1))
            disentangled['bah2bbh'] = self.dropout(F.softmax(disentangled['bah2bbh'], dim=-1))
            disentangled['bah2pan'] = self.dropout(F.softmax(disentangled['bah2pan'], dim=-1))
            disentangled['bah2oan'] = self.dropout(F.softmax(disentangled['bah2oan'], dim=-1))

            disentangled['bbh2a'] = self.dropout(F.softmax(disentangled['bbh2a'], dim=-1))
            disentangled['bbh2s'] = self.dropout(F.softmax(disentangled['bbh2s'], dim=-1))
            disentangled['bbh2bh'] = self.dropout(F.softmax(disentangled['bbh2bh'], dim=-1))
            disentangled['bbh2bah'] = self.dropout(F.softmax(disentangled['bbh2bah'], dim=-1))
            disentangled['bbh2bbh'] = self.dropout(F.softmax(disentangled['bbh2bbh'], dim=-1))
            disentangled['bbh2pan'] = self.dropout(F.softmax(disentangled['bbh2pan'], dim=-1))
            disentangled['bbh2oan'] = self.dropout(F.softmax(disentangled['bbh2oan'], dim=-1))

            disentangled['pan2a'] = self.dropout(F.softmax(disentangled['pan2a'], dim=-1))
            disentangled['pan2s'] = self.dropout(F.softmax(disentangled['pan2s'], dim=-1))
            disentangled['pan2bh'] = self.dropout(F.softmax(disentangled['pan2bh'], dim=-1))
            disentangled['pan2bah'] = self.dropout(F.softmax(disentangled['pan2bah'], dim=-1))
            disentangled['pan2bbh'] = self.dropout(F.softmax(disentangled['pan2bbh'], dim=-1))
            disentangled['pan2pan'] = self.dropout(F.softmax(disentangled['pan2pan'], dim=-1))
            disentangled['pan2oan'] = self.dropout(F.softmax(disentangled['pan2oan'], dim=-1))

            disentangled['oan2a'] = self.dropout(F.softmax(disentangled['oan2a'], dim=-1))
            disentangled['oan2s'] = self.dropout(F.softmax(disentangled['oan2s'], dim=-1))
            disentangled['oan2bh'] = self.dropout(F.softmax(disentangled['oan2bh'], dim=-1))
            disentangled['oan2bah'] = self.dropout(F.softmax(disentangled['oan2bah'], dim=-1))
            disentangled['oan2bbh'] = self.dropout(F.softmax(disentangled['oan2bbh'], dim=-1))
            disentangled['oan2pan'] = self.dropout(F.softmax(disentangled['oan2pan'], dim=-1))
            disentangled['oan2oan'] = self.dropout(F.softmax(disentangled['oan2oan'], dim=-1))

        attention_score = self.dropout(F.softmax(attention_score, dim=-1))
        output = torch.matmul(attention_score, (v_a + v_s + v_bh + v_bah + v_bbh + v_pan + v_oan))

        return output, attention_score, disentangled
class TypeAreaScaledDotProductAttention_Big2(nn.Module):
    ''' Scaled Dot-Product Attention of type-area attention'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, q_bh, k_bh, v_bh, q_bah, k_bah, v_bah, q_bbh, k_bbh, v_bbh, q_pan, k_pan, v_pan, q_oan, k_oan, v_oan, q_pl, k_pl, v_pl, q_ol, k_ol, v_ol,mask=None, return_attns=None):
        a2a = torch.matmul(q_a, k_a.transpose(2, 3))
        a2s = torch.matmul(q_a, k_s.transpose(2, 3))
        a2bh = torch.matmul(q_a, k_bh.transpose(2, 3))
        a2bah = torch.matmul(q_a, k_bah.transpose(2, 3))
        a2bbh = torch.matmul(q_a, k_bbh.transpose(2, 3))
        a2pan = torch.matmul(q_a, k_pan.transpose(2, 3))
        a2oan = torch.matmul(q_a, k_oan.transpose(2, 3))
        a2pl = torch.matmul(q_a, k_pl.transpose(2, 3))
        a2ol = torch.matmul(q_a, k_ol.transpose(2, 3))

        s2a = torch.matmul(q_s, k_a.transpose(2, 3))
        s2s = torch.matmul(q_s, k_s.transpose(2, 3))
        s2bh = torch.matmul(q_s, k_bh.transpose(2, 3))
        s2bah = torch.matmul(q_s, k_bah.transpose(2, 3))
        s2bbh = torch.matmul(q_s, k_bbh.transpose(2, 3))
        s2pan = torch.matmul(q_s, k_pan.transpose(2, 3))
        s2oan = torch.matmul(q_s, k_oan.transpose(2, 3))
        s2pl = torch.matmul(q_s, k_pl.transpose(2, 3))
        s2ol = torch.matmul(q_s, k_ol.transpose(2, 3))

        bh2a = torch.matmul(q_bh, k_a.transpose(2, 3))
        bh2s = torch.matmul(q_bh, k_s.transpose(2, 3))
        bh2bh = torch.matmul(q_bh, k_bh.transpose(2, 3))
        bh2bah = torch.matmul(q_bh, k_bah.transpose(2, 3))
        bh2bbh = torch.matmul(q_bh, k_bbh.transpose(2, 3))
        bh2pan = torch.matmul(q_bh, k_pan.transpose(2, 3))
        bh2oan = torch.matmul(q_bh, k_oan.transpose(2, 3))
        bh2pl = torch.matmul(q_bh, k_pl.transpose(2, 3))
        bh2ol = torch.matmul(q_bh, k_ol.transpose(2, 3))

        bah2a = torch.matmul(q_bah, k_a.transpose(2, 3))
        bah2s = torch.matmul(q_bah, k_s.transpose(2, 3))
        bah2bh = torch.matmul(q_bah, k_bh.transpose(2, 3))
        bah2bah = torch.matmul(q_bah, k_bah.transpose(2, 3))
        bah2bbh = torch.matmul(q_bah, k_bbh.transpose(2, 3))
        bah2pan = torch.matmul(q_bah, k_pan.transpose(2, 3))
        bah2oan = torch.matmul(q_bah, k_oan.transpose(2, 3))
        bah2pl = torch.matmul(q_bah, k_pl.transpose(2, 3))
        bah2ol = torch.matmul(q_bah, k_ol.transpose(2, 3))

        bbh2a = torch.matmul(q_bbh, k_a.transpose(2, 3))
        bbh2s = torch.matmul(q_bbh, k_s.transpose(2, 3))
        bbh2bh = torch.matmul(q_bbh, k_bh.transpose(2, 3))
        bbh2bah = torch.matmul(q_bbh, k_bah.transpose(2, 3))
        bbh2bbh = torch.matmul(q_bbh, k_bbh.transpose(2, 3))
        bbh2pan = torch.matmul(q_bbh, k_pan.transpose(2, 3))
        bbh2oan = torch.matmul(q_bbh, k_oan.transpose(2, 3))
        bbh2pl = torch.matmul(q_bbh, k_pl.transpose(2, 3))
        bbh2ol = torch.matmul(q_bbh, k_ol.transpose(2, 3))

        pan2a = torch.matmul(q_pan, k_a.transpose(2, 3))
        pan2s = torch.matmul(q_pan, k_s.transpose(2, 3))
        pan2bh = torch.matmul(q_pan, k_bh.transpose(2, 3))
        pan2bah = torch.matmul(q_pan, k_bah.transpose(2, 3))
        pan2bbh = torch.matmul(q_pan, k_bbh.transpose(2, 3))
        pan2pan = torch.matmul(q_pan, k_pan.transpose(2, 3))
        pan2oan = torch.matmul(q_pan, k_oan.transpose(2, 3))
        pan2pl = torch.matmul(q_pan, k_pl.transpose(2, 3))
        pan2ol = torch.matmul(q_pan, k_ol.transpose(2, 3))

        oan2a = torch.matmul(q_oan, k_a.transpose(2, 3))
        oan2s = torch.matmul(q_oan, k_s.transpose(2, 3))
        oan2bh = torch.matmul(q_oan, k_bh.transpose(2, 3))
        oan2bah = torch.matmul(q_oan, k_bah.transpose(2, 3))
        oan2bbh = torch.matmul(q_oan, k_bbh.transpose(2, 3))
        oan2pan = torch.matmul(q_oan, k_pan.transpose(2, 3))
        oan2oan = torch.matmul(q_oan, k_oan.transpose(2, 3))
        oan2pl = torch.matmul(q_oan, k_pl.transpose(2, 3))
        oan2ol = torch.matmul(q_oan, k_ol.transpose(2, 3))

        pl2a = torch.matmul(q_pl, k_a.transpose(2, 3))
        pl2s = torch.matmul(q_pl, k_s.transpose(2, 3))
        pl2bh = torch.matmul(q_pl, k_bh.transpose(2, 3))
        pl2bah = torch.matmul(q_pl, k_bah.transpose(2, 3))
        pl2bbh = torch.matmul(q_pl, k_bbh.transpose(2, 3))
        pl2pan = torch.matmul(q_pl, k_pan.transpose(2, 3))
        pl2oan = torch.matmul(q_pl, k_oan.transpose(2, 3))
        pl2pl = torch.matmul(q_pl, k_pl.transpose(2, 3))
        pl2ol = torch.matmul(q_pl, k_ol.transpose(2, 3))

        ol2a = torch.matmul(q_ol, k_a.transpose(2, 3))
        ol2s = torch.matmul(q_ol, k_s.transpose(2, 3))
        ol2bh = torch.matmul(q_ol, k_bh.transpose(2, 3))
        ol2bah = torch.matmul(q_ol, k_bah.transpose(2, 3))
        ol2bbh = torch.matmul(q_ol, k_bbh.transpose(2, 3))
        ol2pan = torch.matmul(q_ol, k_pan.transpose(2, 3))
        ol2oan = torch.matmul(q_ol, k_oan.transpose(2, 3))
        ol2pl = torch.matmul(q_ol, k_pl.transpose(2, 3))
        ol2ol = torch.matmul(q_ol, k_ol.transpose(2, 3))

        attention_score = (a2a + a2s + a2bh + a2bah + a2bbh + a2pan + a2oan + a2pl + a2ol +
                           s2a + s2s + s2bh + s2bah + s2bbh + s2pan + s2oan + s2pl + s2ol +
                           bh2a + bh2s + bh2bh + bh2bah + bh2bbh + bh2pan + bh2oan + bh2pl + bh2ol +
                           bah2a + bah2s + bah2bh + bah2bah + bah2bbh + bah2pan + bah2oan + bah2pl + bah2ol +
                           bbh2a + bbh2s + bbh2bh + bbh2bah + bbh2bbh + bbh2pan + bbh2oan + bbh2pl + bbh2ol +
                           pan2a + pan2s + pan2bh + pan2bah + pan2bbh + pan2pan + pan2oan + pan2pl + pan2ol +
                           oan2a + oan2s + oan2bh + oan2bah + oan2bbh + oan2pan + oan2oan + oan2pl + oan2ol +
                           pl2a + pl2s + pl2bh + pl2bah + pl2bbh + pl2pan + pl2oan + pl2pl + pl2ol +
                           ol2a + ol2s + ol2bh + ol2bah + ol2bbh + ol2pan + ol2oan + ol2pl + ol2ol) / self.temperature

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        disentangled = {}
        if return_attns is not None:
            if mask is not None:
                disentangled['a2a'] = (a2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2s'] = (a2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2bh'] = (a2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2bah'] = (a2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2bbh'] = (a2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2pan'] = (a2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2oan'] = (a2oan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2pl'] = (a2pl / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['a2ol'] = (a2ol / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['s2a'] = (s2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2s'] = (s2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2bh'] = (s2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2bah'] = (s2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2bbh'] = (s2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2pan'] = (s2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2oan'] = (s2oan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2pl'] = (s2pl / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['s2ol'] = (s2ol / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['bh2a'] = (bh2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2s'] = (bh2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2bh'] = (bh2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2bah'] = (bh2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2bbh'] = (bh2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2pan'] = (bh2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2oan'] = (bh2oan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2pl'] = (bh2pl / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bh2ol'] = (bh2ol / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['bah2a'] = (bah2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2s'] = (bah2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2bh'] = (bah2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2bah'] = (bah2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2bbh'] = (bah2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2pan'] = (bah2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2oan'] = (bah2oan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2pl'] = (bah2pl / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bah2ol'] = (bah2ol / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['bbh2a'] = (bbh2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2s'] = (bbh2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2bh'] = (bbh2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2bah'] = (bbh2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2bbh'] = (bbh2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2pan'] = (bbh2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2oan'] = (bbh2oan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2pl'] = (bbh2pl / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['bbh2ol'] = (bbh2ol / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['pan2a'] = (pan2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2s'] = (pan2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2bh'] = (pan2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2bah'] = (pan2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2bbh'] = (pan2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2pan'] = (pan2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2oan'] = (pan2oan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2pl'] = (pan2pl / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pan2ol'] = (pan2ol / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['oan2a'] = (oan2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2s'] = (oan2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2bh'] = (oan2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2bah'] = (oan2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2bbh'] = (oan2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2pan'] = (oan2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2oan'] = (oan2oan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2pl'] = (oan2pl / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['oan2ol'] = (oan2ol / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['pl2a'] = (pl2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pl2s'] = (pl2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pl2bh'] = (pl2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pl2bah'] = (pl2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pl2bbh'] = (pl2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pl2pan'] = (pl2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pl2oan'] = (pl2oan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pl2pl'] = (pl2pl / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['pl2ol'] = (pl2ol / self.temperature).masked_fill(mask == 0, -1e9)

                disentangled['ol2a'] = (ol2a / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['ol2s'] = (ol2s / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['ol2bh'] = (ol2bh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['ol2bah'] = (ol2bah / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['ol2bbh'] = (ol2bbh / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['ol2pan'] = (ol2pan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['ol2oan'] = (ol2oan / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['ol2pl'] = (ol2pl / self.temperature).masked_fill(mask == 0, -1e9)
                disentangled['ol2ol'] = (ol2ol / self.temperature).masked_fill(mask == 0, -1e9)

            else:
                disentangled['a2a'] = (a2a / self.temperature)
                disentangled['a2s'] = (a2s / self.temperature)
                disentangled['a2bh'] = (a2bh / self.temperature)
                disentangled['a2bah'] = (a2bah / self.temperature)
                disentangled['a2bbh'] = (a2bbh / self.temperature)
                disentangled['a2pan'] = (a2pan / self.temperature)
                disentangled['a2oan'] = (a2oan / self.temperature)
                disentangled['a2pl'] = (a2pl / self.temperature)
                disentangled['a2ol'] = (a2ol / self.temperature)

                disentangled['s2a'] = (s2a / self.temperature)
                disentangled['s2s'] = (s2s / self.temperature)
                disentangled['s2bh'] = (s2bh / self.temperature)
                disentangled['s2bah'] = (s2bah / self.temperature)
                disentangled['s2bbh'] = (s2bbh / self.temperature)
                disentangled['s2pan'] = (s2pan / self.temperature)
                disentangled['s2oan'] = (s2oan / self.temperature)
                disentangled['s2pl'] = (s2pl / self.temperature)
                disentangled['s2ol'] = (s2ol / self.temperature)

                disentangled['bh2a'] = (bh2a / self.temperature)
                disentangled['bh2s'] = (bh2s / self.temperature)
                disentangled['bh2bh'] = (bh2bh / self.temperature)
                disentangled['bh2bah'] = (bh2bah / self.temperature)
                disentangled['bh2bbh'] = (bh2bbh / self.temperature)
                disentangled['bh2pan'] = (bh2pan / self.temperature)
                disentangled['bh2oan'] = (bh2oan / self.temperature)
                disentangled['bh2pl'] = (bh2pl / self.temperature)
                disentangled['bh2ol'] = (bh2ol / self.temperature)

                disentangled['bah2a'] = (bah2a / self.temperature)
                disentangled['bah2s'] = (bah2s / self.temperature)
                disentangled['bah2bh'] = (bah2bh / self.temperature)
                disentangled['bah2bah'] = (bah2bah / self.temperature)
                disentangled['bah2bbh'] = (bah2bbh / self.temperature)
                disentangled['bah2pan'] = (bah2pan / self.temperature)
                disentangled['bah2oan'] = (bah2oan / self.temperature)
                disentangled['bah2pl'] = (bah2pl / self.temperature)
                disentangled['bah2ol'] = (bah2ol / self.temperature)

                disentangled['bbh2a'] = (bbh2a / self.temperature)
                disentangled['bbh2s'] = (bbh2s / self.temperature)
                disentangled['bbh2bh'] = (bbh2bh / self.temperature)
                disentangled['bbh2bah'] = (bbh2bah / self.temperature)
                disentangled['bbh2bbh'] = (bbh2bbh / self.temperature)
                disentangled['bbh2pan'] = (bbh2pan / self.temperature)
                disentangled['bbh2oan'] = (bbh2oan / self.temperature)
                disentangled['bbh2pl'] = (bbh2pl / self.temperature)
                disentangled['bbh2ol'] = (bbh2ol / self.temperature)

                disentangled['pan2a'] = (pan2a / self.temperature)
                disentangled['pan2s'] = (pan2s / self.temperature)
                disentangled['pan2bh'] = (pan2bh / self.temperature)
                disentangled['pan2bah'] = (pan2bah / self.temperature)
                disentangled['pan2bbh'] = (pan2bbh / self.temperature)
                disentangled['pan2pan'] = (pan2pan / self.temperature)
                disentangled['pan2oan'] = (pan2oan / self.temperature)
                disentangled['pan2pl'] = (pan2pl / self.temperature)
                disentangled['pan2ol'] = (pan2ol / self.temperature)
                
                disentangled['oan2a'] = (oan2a / self.temperature)
                disentangled['oan2s'] = (oan2s / self.temperature)
                disentangled['oan2bh'] = (oan2bh / self.temperature)
                disentangled['oan2bah'] = (oan2bah / self.temperature)
                disentangled['oan2bbh'] = (oan2bbh / self.temperature)
                disentangled['oan2pan'] = (oan2pan / self.temperature)
                disentangled['oan2oan'] = (oan2oan / self.temperature)
                disentangled['oan2pl'] = (oan2pl / self.temperature)
                disentangled['oan2ol'] = (oan2ol / self.temperature)

                disentangled['pl2a'] = (pl2a / self.temperature)
                disentangled['pl2s'] = (pl2s / self.temperature)
                disentangled['pl2bh'] = (pl2bh / self.temperature)
                disentangled['pl2bah'] = (pl2bah / self.temperature)
                disentangled['pl2bbh'] = (pl2bbh / self.temperature)
                disentangled['pl2pan'] = (pl2pan / self.temperature)
                disentangled['pl2oan'] = (pl2oan / self.temperature)
                disentangled['pl2pl'] = (pl2pl / self.temperature)
                disentangled['pl2ol'] = (pl2ol / self.temperature)

                disentangled['ol2a'] = (ol2a / self.temperature)
                disentangled['ol2s'] = (ol2s / self.temperature)
                disentangled['ol2bh'] = (ol2bh / self.temperature)
                disentangled['ol2bah'] = (ol2bah / self.temperature)
                disentangled['ol2bbh'] = (ol2bbh / self.temperature)
                disentangled['ol2pan'] = (ol2pan / self.temperature)
                disentangled['ol2oan'] = (ol2oan / self.temperature)
                disentangled['ol2pl'] = (ol2pl / self.temperature)
                disentangled['ol2ol'] = (ol2ol / self.temperature)

            disentangled['a2a'] = self.dropout(F.softmax(disentangled['a2a'], dim=-1))
            disentangled['a2s'] = self.dropout(F.softmax(disentangled['a2s'], dim=-1))
            disentangled['a2bh'] = self.dropout(F.softmax(disentangled['a2bh'], dim=-1))
            disentangled['a2bah'] = self.dropout(F.softmax(disentangled['a2bah'], dim=-1))
            disentangled['a2bbh'] = self.dropout(F.softmax(disentangled['a2bbh'], dim=-1))
            disentangled['a2pan'] = self.dropout(F.softmax(disentangled['a2pan'], dim=-1))
            disentangled['a2oan'] = self.dropout(F.softmax(disentangled['a2oan'], dim=-1))
            disentangled['a2pl'] = self.dropout(F.softmax(disentangled['a2pl'], dim=-1))
            disentangled['a2ol'] = self.dropout(F.softmax(disentangled['a2ol'], dim=-1))

            disentangled['s2a'] = self.dropout(F.softmax(disentangled['s2a'], dim=-1))
            disentangled['s2s'] = self.dropout(F.softmax(disentangled['s2s'], dim=-1))
            disentangled['s2bh'] = self.dropout(F.softmax(disentangled['s2bh'], dim=-1))
            disentangled['s2bah'] = self.dropout(F.softmax(disentangled['s2bah'], dim=-1))
            disentangled['s2bbh'] = self.dropout(F.softmax(disentangled['s2bbh'], dim=-1))
            disentangled['s2pan'] = self.dropout(F.softmax(disentangled['s2pan'], dim=-1))
            disentangled['s2oan'] = self.dropout(F.softmax(disentangled['s2oan'], dim=-1))
            disentangled['s2pl'] = self.dropout(F.softmax(disentangled['s2pl'], dim=-1))
            disentangled['s2ol'] = self.dropout(F.softmax(disentangled['s2ol'], dim=-1))

            disentangled['bh2a'] = self.dropout(F.softmax(disentangled['bh2a'], dim=-1))
            disentangled['bh2s'] = self.dropout(F.softmax(disentangled['bh2s'], dim=-1))
            disentangled['bh2bh'] = self.dropout(F.softmax(disentangled['bh2bh'], dim=-1))
            disentangled['bh2bah'] = self.dropout(F.softmax(disentangled['bh2bah'], dim=-1))
            disentangled['bh2bbh'] = self.dropout(F.softmax(disentangled['bh2bbh'], dim=-1))
            disentangled['bh2pan'] = self.dropout(F.softmax(disentangled['bh2pan'], dim=-1))
            disentangled['bh2oan'] = self.dropout(F.softmax(disentangled['bh2oan'], dim=-1))
            disentangled['bh2pl'] = self.dropout(F.softmax(disentangled['bh2pl'], dim=-1))
            disentangled['bh2ol'] = self.dropout(F.softmax(disentangled['bh2ol'], dim=-1))

            disentangled['bah2a'] = self.dropout(F.softmax(disentangled['bah2a'], dim=-1))
            disentangled['bah2s'] = self.dropout(F.softmax(disentangled['bah2s'], dim=-1))
            disentangled['bah2bh'] = self.dropout(F.softmax(disentangled['bah2bh'], dim=-1))
            disentangled['bah2bah'] = self.dropout(F.softmax(disentangled['bah2bah'], dim=-1))
            disentangled['bah2bbh'] = self.dropout(F.softmax(disentangled['bah2bbh'], dim=-1))
            disentangled['bah2pan'] = self.dropout(F.softmax(disentangled['bah2pan'], dim=-1))
            disentangled['bah2oan'] = self.dropout(F.softmax(disentangled['bah2oan'], dim=-1))
            disentangled['bah2pl'] = self.dropout(F.softmax(disentangled['bah2pl'], dim=-1))
            disentangled['bah2ol'] = self.dropout(F.softmax(disentangled['bah2ol'], dim=-1))

            disentangled['bbh2a'] = self.dropout(F.softmax(disentangled['bbh2a'], dim=-1))
            disentangled['bbh2s'] = self.dropout(F.softmax(disentangled['bbh2s'], dim=-1))
            disentangled['bbh2bh'] = self.dropout(F.softmax(disentangled['bbh2bh'], dim=-1))
            disentangled['bbh2bah'] = self.dropout(F.softmax(disentangled['bbh2bah'], dim=-1))
            disentangled['bbh2bbh'] = self.dropout(F.softmax(disentangled['bbh2bbh'], dim=-1))
            disentangled['bbh2pan'] = self.dropout(F.softmax(disentangled['bbh2pan'], dim=-1))
            disentangled['bbh2oan'] = self.dropout(F.softmax(disentangled['bbh2oan'], dim=-1))
            disentangled['bbh2pl'] = self.dropout(F.softmax(disentangled['bbh2pl'], dim=-1))
            disentangled['bbh2ol'] = self.dropout(F.softmax(disentangled['bbh2ol'], dim=-1))

            disentangled['pan2a'] = self.dropout(F.softmax(disentangled['pan2a'], dim=-1))
            disentangled['pan2s'] = self.dropout(F.softmax(disentangled['pan2s'], dim=-1))
            disentangled['pan2bh'] = self.dropout(F.softmax(disentangled['pan2bh'], dim=-1))
            disentangled['pan2bah'] = self.dropout(F.softmax(disentangled['pan2bah'], dim=-1))
            disentangled['pan2bbh'] = self.dropout(F.softmax(disentangled['pan2bbh'], dim=-1))
            disentangled['pan2pan'] = self.dropout(F.softmax(disentangled['pan2pan'], dim=-1))
            disentangled['pan2oan'] = self.dropout(F.softmax(disentangled['pan2oan'], dim=-1))
            disentangled['pan2pl'] = self.dropout(F.softmax(disentangled['pan2pl'], dim=-1))
            disentangled['pan2ol'] = self.dropout(F.softmax(disentangled['pan2ol'], dim=-1))

            disentangled['oan2a'] = self.dropout(F.softmax(disentangled['oan2a'], dim=-1))
            disentangled['oan2s'] = self.dropout(F.softmax(disentangled['oan2s'], dim=-1))
            disentangled['oan2bh'] = self.dropout(F.softmax(disentangled['oan2bh'], dim=-1))
            disentangled['oan2bah'] = self.dropout(F.softmax(disentangled['oan2bah'], dim=-1))
            disentangled['oan2bbh'] = self.dropout(F.softmax(disentangled['oan2bbh'], dim=-1))
            disentangled['oan2pan'] = self.dropout(F.softmax(disentangled['oan2pan'], dim=-1))
            disentangled['oan2oan'] = self.dropout(F.softmax(disentangled['oan2oan'], dim=-1))
            disentangled['oan2pl'] = self.dropout(F.softmax(disentangled['oan2pl'], dim=-1))
            disentangled['oan2ol'] = self.dropout(F.softmax(disentangled['oan2ol'], dim=-1))

            disentangled['pl2a'] = self.dropout(F.softmax(disentangled['pl2a'], dim=-1))
            disentangled['pl2s'] = self.dropout(F.softmax(disentangled['pl2s'], dim=-1))
            disentangled['pl2bh'] = self.dropout(F.softmax(disentangled['pl2bh'], dim=-1))
            disentangled['pl2bah'] = self.dropout(F.softmax(disentangled['pl2bah'], dim=-1))
            disentangled['pl2bbh'] = self.dropout(F.softmax(disentangled['pl2bbh'], dim=-1))
            disentangled['pl2pan'] = self.dropout(F.softmax(disentangled['pl2pan'], dim=-1))
            disentangled['pl2oan'] = self.dropout(F.softmax(disentangled['pl2oan'], dim=-1))
            disentangled['pl2pl'] = self.dropout(F.softmax(disentangled['pl2pl'], dim=-1))
            disentangled['pl2ol'] = self.dropout(F.softmax(disentangled['pl2ol'], dim=-1))

            disentangled['ol2a'] = self.dropout(F.softmax(disentangled['ol2a'], dim=-1))
            disentangled['ol2s'] = self.dropout(F.softmax(disentangled['ol2s'], dim=-1))
            disentangled['ol2bh'] = self.dropout(F.softmax(disentangled['ol2bh'], dim=-1))
            disentangled['ol2bah'] = self.dropout(F.softmax(disentangled['ol2bah'], dim=-1))
            disentangled['ol2bbh'] = self.dropout(F.softmax(disentangled['ol2bbh'], dim=-1))
            disentangled['ol2pan'] = self.dropout(F.softmax(disentangled['ol2pan'], dim=-1))
            disentangled['ol2oan'] = self.dropout(F.softmax(disentangled['ol2oan'], dim=-1))
            disentangled['ol2pl'] = self.dropout(F.softmax(disentangled['ol2pl'], dim=-1))
            disentangled['ol2ol'] = self.dropout(F.softmax(disentangled['ol2ol'], dim=-1))

            

        attention_score = self.dropout(F.softmax(attention_score, dim=-1))
        output = torch.matmul(attention_score, (v_a + v_s + v_bh + v_bah + v_bbh + v_pan + v_oan + v_pl + v_ol))

        return output, attention_score, disentangled
    
class TypeAreaMultiHeadAttention(nn.Module):
    ''' Multi-Head Type-Area Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qa = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ka = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_va = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        scaling_factor = (4 * d_k) ** 0.5

        self.attention = TypeAreaScaledDotProductAttention(temperature=scaling_factor)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, mask=None, return_attns=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q_a.size(0), q_a.size(1), k_a.size(1), v_a.size(1)

        residual_a = q_a
        residual_s = q_s

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_a = self.w_qa(q_a).view(sz_b, len_q, n_head, d_k)
        k_a = self.w_ka(k_a).view(sz_b, len_k, n_head, d_k)
        v_a = self.w_va(v_a).view(sz_b, len_v, n_head, d_v)

        q_s = self.w_qs(q_s).view(sz_b, len_q, n_head, d_k)
        k_s = self.w_ks(k_s).view(sz_b, len_k, n_head, d_k)
        v_s = self.w_vs(v_s).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q_a, k_a, v_a = q_a.transpose(1, 2), k_a.transpose(1, 2), v_a.transpose(1, 2)
        q_s, k_s, v_s = q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        output, attn, disentangled = self.attention(q_a, k_a, v_a, q_s, k_s, v_s, mask=mask, return_attns=return_attns)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))

        output += (residual_a + residual_s)
        output = self.layer_norm(output)

        return output, attn, disentangled
class TypeAreaMultiHeadAttention_Big(nn.Module):
    ''' Multi-Head Type-Area Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qa = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ka = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_va = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qbh = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kbh = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vbh = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qbah = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kbah = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vbah = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qbbh = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kbbh = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vbbh = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qpan = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kpan = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vpan = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qoan = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_koan = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_voan = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        scaling_factor = (4 * d_k) ** 0.5

        self.attention = TypeAreaScaledDotProductAttention_Big(temperature=scaling_factor)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, q_bh, k_bh, v_bh, q_bah, k_bah, v_bah, q_bbh, k_bbh, v_bbh, q_pan, k_pan, v_pan, q_oan, k_oan, v_oan, mask=None, return_attns=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q_a.size(0), q_a.size(1), k_a.size(1), v_a.size(1)

        residual_a = q_a
        residual_s = q_s
        residual_bh = q_bh
        residual_bah = q_bah
        residual_bbh = q_bbh
        residual_pan = q_pan
        residual_oan = q_oan

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_a = self.w_qa(q_a).view(sz_b, len_q, n_head, d_k)
        k_a = self.w_ka(k_a).view(sz_b, len_k, n_head, d_k)
        v_a = self.w_va(v_a).view(sz_b, len_v, n_head, d_v)

        q_s = self.w_qs(q_s).view(sz_b, len_q, n_head, d_k)
        k_s = self.w_ks(k_s).view(sz_b, len_k, n_head, d_k)
        v_s = self.w_vs(v_s).view(sz_b, len_v, n_head, d_v)

        q_bh = self.w_qbh(q_bh).view(sz_b, len_q, n_head, d_k)
        k_bh = self.w_kbh(k_bh).view(sz_b, len_k, n_head, d_k)
        v_bh = self.w_vbh(v_bh).view(sz_b, len_v, n_head, d_v)

        q_bah = self.w_qbah(q_bah).view(sz_b, len_q, n_head, d_k)
        k_bah = self.w_kbah(k_bah).view(sz_b, len_k, n_head, d_k)
        v_bah = self.w_vbah(v_bah).view(sz_b, len_v, n_head, d_v)

        q_bbh = self.w_qbbh(q_bbh).view(sz_b, len_q, n_head, d_k)
        k_bbh = self.w_kbbh(k_bbh).view(sz_b, len_k, n_head, d_k)
        v_bbh = self.w_vbbh(v_bbh).view(sz_b, len_v, n_head, d_v)

        q_pan = self.w_qpan(q_pan).view(sz_b, len_q, n_head, d_k)
        k_pan = self.w_kpan(k_pan).view(sz_b, len_k, n_head, d_k)
        v_pan = self.w_vpan(v_pan).view(sz_b, len_v, n_head, d_v)

        q_oan = self.w_qoan(q_oan).view(sz_b, len_q, n_head, d_k)
        k_oan = self.w_koan(k_oan).view(sz_b, len_k, n_head, d_k)
        v_oan = self.w_voan(v_oan).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q_a, k_a, v_a = q_a.transpose(1, 2), k_a.transpose(1, 2), v_a.transpose(1, 2)
        q_s, k_s, v_s = q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2)
        q_bh, k_bh, v_bh = q_bh.transpose(1, 2), k_bh.transpose(1, 2), v_bh.transpose(1, 2)
        q_bah, k_bah, v_bah = q_bah.transpose(1, 2), k_bah.transpose(1, 2), v_bah.transpose(1, 2)
        q_bbh, k_bbh, v_bbh = q_bbh.transpose(1, 2), k_bbh.transpose(1, 2), v_bbh.transpose(1, 2)
        q_pan, k_pan, v_pan = q_pan.transpose(1, 2), k_pan.transpose(1, 2), v_pan.transpose(1, 2)
        q_oan, k_oan, v_oan = q_oan.transpose(1, 2), k_oan.transpose(1, 2), v_oan.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        output, attn, disentangled = self.attention(q_a, k_a, v_a, q_s, k_s, v_s, q_bh, k_bh, v_bh, q_bah, k_bah, v_bah,
                                                    q_bbh, k_bbh, v_bbh, q_pan, k_pan, v_pan, q_oan, k_oan, v_oan, mask=mask, return_attns=return_attns)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))

        output += (residual_a + residual_s + residual_bh + residual_bah + residual_bbh + residual_pan + residual_oan)
        output = self.layer_norm(output)

        return output, attn, disentangled
class TypeAreaMultiHeadAttention_Big2(nn.Module):
    ''' Multi-Head Type-Area Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qa = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ka = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_va = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qbh = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kbh = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vbh = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qbah = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kbah = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vbah = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qbbh = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kbbh = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vbbh = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qpan = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kpan = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vpan = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qoan = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_koan = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_voan = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qpl = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kpl = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vpl = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qol = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kol = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vol = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        scaling_factor = (4 * d_k) ** 0.5

        self.attention = TypeAreaScaledDotProductAttention_Big2(temperature=scaling_factor)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, q_bh, k_bh, v_bh, q_bah, k_bah, v_bah, q_bbh, k_bbh, v_bbh, q_pan, k_pan, v_pan, q_oan, k_oan, v_oan, q_pl, k_pl, v_pl, q_ol, k_ol, v_ol,mask=None, return_attns=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q_a.size(0), q_a.size(1), k_a.size(1), v_a.size(1)

        residual_a = q_a
        residual_s = q_s
        residual_bh = q_bh
        residual_bah = q_bah
        residual_bbh = q_bbh
        residual_pan = q_pan
        residual_oan = q_oan
        residual_pl = q_pl
        residual_ol = q_ol

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_a = self.w_qa(q_a).view(sz_b, len_q, n_head, d_k)
        k_a = self.w_ka(k_a).view(sz_b, len_k, n_head, d_k)
        v_a = self.w_va(v_a).view(sz_b, len_v, n_head, d_v)

        q_s = self.w_qs(q_s).view(sz_b, len_q, n_head, d_k)
        k_s = self.w_ks(k_s).view(sz_b, len_k, n_head, d_k)
        v_s = self.w_vs(v_s).view(sz_b, len_v, n_head, d_v)

        q_bh = self.w_qbh(q_bh).view(sz_b, len_q, n_head, d_k)
        k_bh = self.w_kbh(k_bh).view(sz_b, len_k, n_head, d_k)
        v_bh = self.w_vbh(v_bh).view(sz_b, len_v, n_head, d_v)

        q_bah = self.w_qbah(q_bah).view(sz_b, len_q, n_head, d_k)
        k_bah = self.w_kbah(k_bah).view(sz_b, len_k, n_head, d_k)
        v_bah = self.w_vbah(v_bah).view(sz_b, len_v, n_head, d_v)

        q_bbh = self.w_qbbh(q_bbh).view(sz_b, len_q, n_head, d_k)
        k_bbh = self.w_kbbh(k_bbh).view(sz_b, len_k, n_head, d_k)
        v_bbh = self.w_vbbh(v_bbh).view(sz_b, len_v, n_head, d_v)

        q_pan = self.w_qpan(q_pan).view(sz_b, len_q, n_head, d_k)
        k_pan = self.w_kpan(k_pan).view(sz_b, len_k, n_head, d_k)
        v_pan = self.w_vpan(v_pan).view(sz_b, len_v, n_head, d_v)

        q_oan = self.w_qoan(q_oan).view(sz_b, len_q, n_head, d_k)
        k_oan = self.w_koan(k_oan).view(sz_b, len_k, n_head, d_k)
        v_oan = self.w_voan(v_oan).view(sz_b, len_v, n_head, d_v)

        q_pl = self.w_qpl(q_pl).view(sz_b, len_q, n_head, d_k)
        k_pl = self.w_kpl(k_pl).view(sz_b, len_k, n_head, d_k)
        v_pl = self.w_vpl(v_pl).view(sz_b, len_v, n_head, d_v)

        q_ol = self.w_qol(q_ol).view(sz_b, len_q, n_head, d_k)
        k_ol = self.w_kol(k_ol).view(sz_b, len_k, n_head, d_k)
        v_ol = self.w_vol(v_ol).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q_a, k_a, v_a = q_a.transpose(1, 2), k_a.transpose(1, 2), v_a.transpose(1, 2)
        q_s, k_s, v_s = q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2)
        q_bh, k_bh, v_bh = q_bh.transpose(1, 2), k_bh.transpose(1, 2), v_bh.transpose(1, 2)
        q_bah, k_bah, v_bah = q_bah.transpose(1, 2), k_bah.transpose(1, 2), v_bah.transpose(1, 2)
        q_bbh, k_bbh, v_bbh = q_bbh.transpose(1, 2), k_bbh.transpose(1, 2), v_bbh.transpose(1, 2)
        q_pan, k_pan, v_pan = q_pan.transpose(1, 2), k_pan.transpose(1, 2), v_pan.transpose(1, 2)
        q_oan, k_oan, v_oan = q_oan.transpose(1, 2), k_oan.transpose(1, 2), v_oan.transpose(1, 2)
        q_pl, k_pl, v_pl = q_pl.transpose(1, 2), k_pl.transpose(1, 2), v_pl.transpose(1, 2)
        q_ol, k_ol, v_ol = q_ol.transpose(1, 2), k_ol.transpose(1, 2), v_ol.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        output, attn, disentangled = self.attention(q_a, k_a, v_a, q_s, k_s, v_s, q_bh, k_bh, v_bh, q_bah, k_bah, v_bah,
                                                    q_bbh, k_bbh, v_bbh, q_pan, k_pan, v_pan, q_oan, k_oan, v_oan, q_pl, k_pl, v_pl, q_ol, k_ol, v_ol, mask=mask, return_attns=return_attns)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))

        output += (residual_a + residual_s + residual_bh + residual_bah + residual_bbh + residual_pan + residual_oan + residual_pl + residual_ol)
        output = self.layer_norm(output)

        return output, attn, disentangled

class TypeAreaMultiHeadAttention2(nn.Module):
    ''' Multi-Head Type-Area Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qa = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ka = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_va = nn.Linear(d_model, n_head * d_v, bias=False)

        self.w_qr = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kr = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vr = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        scaling_factor = (4 * d_k) ** 0.5

        self.attention = TypeAreaScaledDotProductAttention(temperature=scaling_factor)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, q_bh, k_bh, v_bh, q_bah, k_bah, v_bah, q_bbh, k_bbh, v_bbh, q_pan, k_pan, v_pan, q_oan, k_oan, v_oan, mask=None, return_attns=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q_a.size(0), q_a.size(1), k_a.size(1), v_a.size(1)

        residual_a = q_a
        residual_s = q_s
        residual_bh = q_bh
        residual_bah = q_bah
        residual_bbh = q_bbh
        residual_pan = q_pan
        residual_oan = q_oan

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q_a = self.w_qa(q_a).view(sz_b, len_q, n_head, d_k)
        k_a = self.w_ka(k_a).view(sz_b, len_k, n_head, d_k)
        v_a = self.w_va(v_a).view(sz_b, len_v, n_head, d_v)

        q_s = self.w_qs(q_s).view(sz_b, len_q, n_head, d_k)
        k_s = self.w_ks(k_s).view(sz_b, len_k, n_head, d_k)
        v_s = self.w_vs(v_s).view(sz_b, len_v, n_head, d_v)

        q_bh = self.w_qr(q_bh).view(sz_b, len_q, n_head, d_k)
        k_bh = self.w_kr(k_bh).view(sz_b, len_k, n_head, d_k)
        v_bh = self.w_vr(v_bh).view(sz_b, len_v, n_head, d_v)

        q_bah = self.w_qr(q_bah).view(sz_b, len_q, n_head, d_k)
        k_bah = self.w_kr(k_bah).view(sz_b, len_k, n_head, d_k)
        v_bah = self.w_vr(v_bah).view(sz_b, len_v, n_head, d_v)

        q_bbh = self.w_qr(q_bbh).view(sz_b, len_q, n_head, d_k)
        k_bbh = self.w_kr(k_bbh).view(sz_b, len_k, n_head, d_k)
        v_bbh = self.w_vr(v_bbh).view(sz_b, len_v, n_head, d_v)

        q_pan = self.w_qr(q_pan).view(sz_b, len_q, n_head, d_k)
        k_pan = self.w_kr(k_pan).view(sz_b, len_k, n_head, d_k)
        v_pan = self.w_vr(v_pan).view(sz_b, len_v, n_head, d_v)

        q_oan = self.w_qr(q_oan).view(sz_b, len_q, n_head, d_k)
        k_oan = self.w_kr(k_oan).view(sz_b, len_k, n_head, d_k)
        v_oan = self.w_vr(v_oan).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q_a, k_a, v_a = q_a.transpose(1, 2), k_a.transpose(1, 2), v_a.transpose(1, 2)
        q_s, k_s, v_s = q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2)
        q_bh, k_bh, v_bh = q_bh.transpose(1, 2), k_bh.transpose(1, 2), v_bh.transpose(1, 2)
        q_bah, k_bah, v_bah = q_bah.transpose(1, 2), k_bah.transpose(1, 2), v_bah.transpose(1, 2)
        q_bbh, k_bbh, v_bbh = q_bbh.transpose(1, 2), k_bbh.transpose(1, 2), v_bbh.transpose(1, 2)
        q_pan, k_pan, v_pan = q_pan.transpose(1, 2), k_pan.transpose(1, 2), v_pan.transpose(1, 2)
        q_oan, k_oan, v_oan = q_oan.transpose(1, 2), k_oan.transpose(1, 2), v_oan.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        output, attn, disentangled = self.attention(q_a, k_a, v_a, q_s, k_s, v_s, q_bh, k_bh, v_bh, q_bah, k_bah, v_bah,
                                                    q_bbh, k_bbh, v_bbh, q_pan, k_pan, v_pan, q_oan, k_oan, v_oan, mask=mask, return_attns=return_attns)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))

        output += (residual_a + residual_s + residual_bh + residual_bah + residual_bbh + residual_pan + residual_oan)
        output = self.layer_norm(output)

        return output, attn, disentangled

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x