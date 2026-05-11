import torch

# ── Data ──────────────────────────────────────────────────────────────────────
BLOCK_SIZE   = 256        # context length (tokens per sample)
VOCAB_SIZE   = 5000       # must match tokenizer vocab_size

# ── Model ─────────────────────────────────────────────────────────────────────
N_EMBD       = 768        # embedding dimension
N_HEAD       = 12         # number of attention heads
N_LAYER      = 8          # number of transformer blocks
DROPOUT      = 0.1        # dropout rate

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 128
LR           = 1e-4
MAX_EPOCHS   = 10
EVAL_INTERVAL= 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42