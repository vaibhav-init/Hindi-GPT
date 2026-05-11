import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config as cfg    
import config_task3 as cfg3                                  


def causal_mask(size):
    return torch.tril(torch.ones(size, size))


class MaskedAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.key   = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(cfg.DROPOUT)            

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(K.size(-1))
        mask = causal_mask(T).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)                       

        return weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.heads = nn.ModuleList([
            MaskedAttentionHead(embed_dim, self.head_dim)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForwardNN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),                      
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn  = FeedForwardNN(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.emb_dropout        = nn.Dropout(cfg.DROPOUT)  

        self.blocks   = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos     = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = self.emb_dropout(tok_emb + pos_emb)              
        x = self.blocks(x)
        x = self.ln_final(x)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers):
        super().__init__()
        self.backbone = MiniGPT(vocab_size, embed_dim, block_size, num_heads, num_layers)
        self.lm_head  = nn.Linear(embed_dim, vocab_size, bias=False)


    def forward(self, idx, targets=None):
        x      = self.backbone(idx)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V      = logits.shape
            logits_flat  = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=0                         
            )

        return logits, loss

# For task 3
class GPTClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads, num_layers, num_classes):
        super().__init__()
        self.backbone   = MiniGPT(vocab_size, embed_dim, block_size, num_heads, num_layers)
        self.proj       = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg3.DROPOUT),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.backbone(idx)                               

        # extract last real (non-PAD) token per sample
        lengths = (idx != 0).sum(dim=1) - 1                
        lengths = lengths.clamp(min=0)                      
        x = x[torch.arange(B, device=idx.device), lengths]  

        x      = torch.relu(self.proj(x))
        logits = self.classifier(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1)

        return logits, loss