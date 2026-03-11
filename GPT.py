import torch 
import torch.nn as nn

from transformer import TransformerBlock, LayerNorm

class GPTModel(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embs = self.token_emb(in_idx)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = token_embs + pos_embs
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits


def GenerateText(model, idx, max_new_tokens, context_size):
    
    
    for _ in range(max_new_tokens):
        
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        
        probas = torch.softmax(logits, dim=-1)
        
        next_id = torch.argmax(probas, dim=-1, keepdim=True)
        
        idx = torch.cat((idx, next_id), dim=1)
    
    return idx
