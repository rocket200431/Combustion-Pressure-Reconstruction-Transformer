import torch
import torch.nn as nn
import math

# ---------------- Positional Encoding ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ---------------- Multi-Head Attention ----------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.d = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)

# ---------------- PressureNet ----------------
class PressureNet(nn.Module):
    def __init__(self,
                 d_model=128,
                 heads=4,
                 layers=3,
                 dropout=0.1,
                 mask_start=50,
                 mask_end=60):
        super().__init__()

        self.mask_start = mask_start
        self.mask_end = mask_end

        # NOTE: input dim = 2 (value + mask)
        self.embed = nn.Linear(2, d_model)
        self.pos = PositionalEncoding(d_model)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                MultiHeadSelfAttention(d_model, heads),
                nn.Dropout(dropout)
            )
            for _ in range(layers)
        ])

        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, 100, 2)
        x = self.embed(x)
        x = self.pos(x)

        for block in self.blocks:
            x = x + block(x)

        masked = x[:, self.mask_start:self.mask_end]
        return self.out(masked).squeeze(-1)
