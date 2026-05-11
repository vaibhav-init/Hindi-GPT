import torch

# ── Data ──────────────────────────────────────────────────────────────────────
BLOCK_SIZE   = 512        # larger to avoid truncating long movie reviews
VOCAB_SIZE   = 5000       # must match tokenizer vocab_size

# ── Model ─────────────────────────────────────────────────────────────────────
N_EMBD       = 768        # must match pretrained LM backbone
N_HEAD       = 12         # must match pretrained LM backbone
N_LAYER      = 8          # must match pretrained LM backbone
DROPOUT      = 0.5        # dropout rate

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32         # smaller — only ~646 training samples
LR           = 3e-5       # lower LR for fine-tuning
MAX_EPOCHS   = 15
EVAL_INTERVAL= 500
NUM_CLASSES  = 3          # Negative, Neutral, Positive
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42