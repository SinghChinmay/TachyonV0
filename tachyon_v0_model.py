import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveEngine(nn.Module):
    """TachyonV0の核心エンジン: (w * (x1 * cos(x2)))"""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.02))

    def forward(self, x1, x2):
        return self.w * (x1 * torch.cos(x2))

class TachyonV0Block(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.wave_h = WaveEngine() # 横(時間)の波
        
        self.ln2 = nn.LayerNorm(n_embd)
        self.wave_v = WaveEngine() # 縦(次元)の波

    def forward(self, x):
        B, T, C = x.size()
        
        # 1. 時間軸の連鎖干渉
        res = x
        x_norm = self.ln1(x)
        x_past_t = torch.cat([torch.zeros(B, 1, C, device=x.device), x_norm[:, :-1, :]], dim=1)
        x = res + self.wave_h(x_norm, x_past_t)
        
        # 2. 次元軸の連鎖干渉
        res = x
        x_norm = self.ln2(x)
        x_past_c = torch.roll(x_norm, shifts=1, dims=-1)
        x = res + self.wave_v(x_norm, x_past_c)
        
        return x

class TachyonV0(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=4096, n_layer=64, block_size=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.ModuleList([TachyonV0Block(n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
