import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class CustomRNN(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            batch_first=True,
            max_seq_length=60,
            attention_maxscore=None,
    ):
        super(CustomRNN, self).__init__()
        self.bidirect = False
        self.num_layers = 1
        self.num_heads = 1
        self.batch_first = batch_first
        self.with_weight = False
        self.max_seq_length = max_seq_length
        self.attention_maxscore = attention_maxscore
        encoder_layer = TransformerEncoderLayer(
            d_model=input_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.rnn = TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.pooling = nn.AdaptiveMaxPool2d((1, input_size))

    def forward(self, inputs, seq_lengths, sen_mask, method="LSTM"):
        # input.size = (batch_size, max_seq_length, node_num)
        # method can be "Pool", "LSTM", or 'AttLSTM"
        # === 强制对齐 inputs 和 sen_mask 的长度 ===
        B, L_in, H = inputs.shape

        # === 1. 强制 sen_mask 与 inputs 长度一致 ===
        if sen_mask.size(1) != L_in:
            if sen_mask.size(1) > L_in:
                sen_mask = sen_mask[:, :L_in]
            else:
                # padding sen_mask (rare)
                pad = torch.zeros(B, L_in - sen_mask.size(1), device=sen_mask.device, dtype=sen_mask.dtype)
                sen_mask = torch.cat([sen_mask, pad], dim=1)
        if method == "LSTM":
            key_padding_mask = (sen_mask == 0)  # [B, L_in]
            output = self.rnn(inputs, src_key_padding_mask=key_padding_mask)  # [B, L_in, H]

            # === 3. 用原始 output + 对齐后的 sen_mask 计算 final_rep ===
            masked_output = output * sen_mask.unsqueeze(-1).float()  # [B, L_in, H]
            lengths = sen_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            final_rep = masked_output.sum(dim=1) / lengths  # [B, H]

            # === 4. 单独构造 padded_output（不影响 final_rep）===
            if L_in < self.max_seq_length:
                pad = torch.zeros(B, self.max_seq_length - L_in, H, device=output.device)
                padded_output = torch.cat([output, pad], dim=1)  # [B, max_seq_length, H]
            else:
                padded_output = output[:, :self.max_seq_length]

            return final_rep, padded_output
        elif method == "AttLSTM":
            sen_mask = torch.tensor(
                np.hstack([[[1]] * inputs.size()[0], sen_mask.cpu()])
            ).cuda()
            att_inputs, att_inputs_weight = attention(
                inputs,
                inputs,
                inputs,
                sen_mask,
                attention_maxscore=self.attention_maxscore,
            )
            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
                att_inputs,
                seq_lengths.to("cpu"),
                batch_first=self.batch_first,
                enforce_sorted=False,
            )

            res, (hn, cn) = self.rnn(input=packed_inputs)

            padded_res, _ = nn.utils.rnn.pad_packed_sequence(
                res, batch_first=self.batch_first, total_length=self.max_seq_length
            )
            return hn.squeeze(0), padded_res
        else:
            out = self.pooling(inputs)
            return out.squeeze(1), None


def attention(query, key, value, mask=None, dropout=None, attention_maxscore=1000):
    """Compute scaled dot product attention"""
    d_k = query.size(-1)
    query = f.normalize(query, p=2, dim=-1)
    key = f.normalize(key, p=2, dim=-1)
    scores = (
            torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) * attention_maxscore
    )
    p_attn = None
    if mask is not None:
        for s, m in zip(scores, mask):
            s = s.masked_fill(m == 0, -1e9)
            p = s.softmax(dim=-1)
            if p_attn is None:
                p_attn = p
            else:
                p_attn = torch.cat([p_attn, p], dim=0)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
