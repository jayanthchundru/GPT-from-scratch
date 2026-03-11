import torch 
import torch.nn as nn
from attention import MultiHeadAttention

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.ones(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = ( x - mean) / torch.sqrt( var + self.eps )
        
        return self.scale * norm_x + self.shift 

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * ( 1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi )) * ( x + 0.044715 * torch.pow(x , 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']), 
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']), 
        )
    
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(inp_dim=cfg['emb_dim'], 
                out_dim=cfg['emb_dim'], 
                context_length=cfg['context_length'],
                num_heads=cfg['n_heads'],
                dropout=cfg['drop_rate'],
                qkv_bias=cfg['qkv_bias'])
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_out = nn.Dropout(cfg['drop_rate'])
        
    def forward(self, x):
        
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_out(x)
        x = x + shortcut 
        
        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_out(x)
        x = x + shortcut
        
        return x
