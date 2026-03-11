import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    
    def __init__(self, inp_dim , out_dim, qkv_bias = False):
        super().__init__()
        self.query_weights = nn.Linear(inp_dim, out_dim, bias= qkv_bias)
        self.key_weights = nn.Linear(inp_dim, out_dim, bias= qkv_bias)
        self.value_weights = nn.Linear(inp_dim, out_dim, bias= qkv_bias)
    
    def forward(self, x):
        queries = self.query_weights(x)
        keys = self.key_weights(x)
        values = self.value_weights(x)
        
        attention_scores = queries @ keys.transpose(-2, -1)
        
        attention_weights = torch.softmax( attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        
        context_vectors = attention_weights @ values 
        
        return context_vectors 


class CausalSelfAttention(nn.Module):
    
    def __init__(self, inp_dim , out_dim, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.query_weights = nn.Linear(inp_dim, out_dim, bias= qkv_bias)
        self.key_weights = nn.Linear(inp_dim, out_dim, bias= qkv_bias)
        self.value_weights = nn.Linear(inp_dim, out_dim, bias= qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
        
        
    def forward(self, x):
        
        b, num_tokens, d_in = x.shape
        queries = self.query_weights(x)
        keys = self.key_weights(x)
        values = self.value_weights(x)
        
        attention_scores = queries @ keys.transpose(-2, -1)
        
        attention_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        
        causal_attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5 , dim=-1)
        
        causal_attention_weights = self.dropout(causal_attention_weights)
        
        context_vectors = causal_attention_weights @ values
        
        return context_vectors


class MultiHeadAttentionWrapper(nn.Module):
    
    def __init__(self, inp_dim , out_dim, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList([CausalSelfAttention(inp_dim , out_dim, context_length, dropout, qkv_bias) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.head], dim=-1)


class MultiHeadAttention(nn.Module):
    
    def __init__(self, inp_dim , out_dim, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim should be divisible by num_heads"
        
        self.out_dim = out_dim 
        self.num_heads = num_heads
        self.head_dim = out_dim  // num_heads
        
        self.query_weights = nn.Linear(inp_dim, out_dim, bias= qkv_bias)
        self.key_weights = nn.Linear(inp_dim, out_dim, bias= qkv_bias)
        self.value_weights = nn.Linear(inp_dim, out_dim, bias= qkv_bias)
        
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.query_weights(x)
        keys = self.key_weights(x)
        values = self.value_weights(x)
        
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(-2, -1)
        
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5 , dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        
        ### getting context vectors ( b , num_tokens , num_heads, head_dim)
        
        context_vectors = attn_weights @ values
        
        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.out_dim)
        
        context_vectors = self.out_proj(context_vectors)
        
        return context_vectors

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super.__init__()
    
    def forward(self, x):
        
        return x 

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super.__init__()
    
    def forward(self, x):
        
        return x 


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super.__init__()
        self.token_embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['emb_dim'])
        
        self.trf_block = nn.sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias= False
        )
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = token_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
