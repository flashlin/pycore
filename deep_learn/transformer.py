import copy

from torch import nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F

from pcore.deep_learn.feed_forward import FeedForward, PositionalEncoder, Embedder
from pcore.deep_learn.normalisation import Norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiheadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiheadAttention(heads, d_model)
        self.attn_2 = MultiheadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
                                           src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


def generate_multiple_layers(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n, heads):
        super().__init__()
        self.N = n
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = generate_multiple_layers(EncoderLayer(d_model, heads), n)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n, heads):
        super().__init__()
        self.N = n
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = generate_multiple_layers(DecoderLayer(d_model, heads), n)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, n, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n, heads)
        self.decoder = Decoder(trg_vocab, d_model, n, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

#
# class TransformerNet(nn.Module):
#     trg_pad
#     def __init__(self):
#         d_model = 512
#         heads = 8
#         N = 6
#         src_vocab = len(EN_TEXT.vocab)
#         trg_vocab = len(FR_TEXT.vocab)
#         model = Transformer(src_vocab, trg_vocab, d_model, N, heads)
#         for p in model.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#         #optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#
#     def compute_loss(self, preds, trg):
#         trg = trg.transpose(0, 1)
#         ys = trg[:, 1:].contiguous().view(-1)
#         loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)