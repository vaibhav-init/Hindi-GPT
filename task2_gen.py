import torch
import sys
import os
import sentencepiece as spm
import config as cfg
from task2_model import GPTLanguageModel


def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load("model_hindi.model")
    print("Tokenizer loaded")
    return sp

def load_model():
    model = GPTLanguageModel(
        vocab_size = cfg.VOCAB_SIZE,
        embed_dim  = cfg.N_EMBD,
        block_size = cfg.BLOCK_SIZE,
        num_heads  = cfg.N_HEAD,
        num_layers = cfg.N_LAYER
    ).to(cfg.DEVICE)

    if not os.path.exists("gpt_hindi_best.pt"):
        raise FileNotFoundError("gpt_hindi_best.pt not found. Train Task 2 first.")

    model.load_state_dict(torch.load("gpt_hindi_best.pt", map_location=cfg.DEVICE))
    model.eval()
    print("Model loaded from gpt_hindi_best.pt")
    return model



def generate(model, sp, prompt, max_new_tokens=50):
    bos_id    = sp.piece_to_id("<BOS>")
    input_ids = torch.tensor(
        [[bos_id] + sp.encode(prompt, out_type=int)],
        dtype=torch.long
    ).to(cfg.DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond   = input_ids[:, -cfg.BLOCK_SIZE:]
            logits, _  = model(idx_cond)
            logits     = logits[:, -1, :]
            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids  = torch.cat([input_ids, next_token], dim=1)

    # strip BOS token before decoding
    tokens = input_ids[0].tolist()
    bos_positions = [i for i, t in enumerate(tokens) if t == bos_id]
    start = bos_positions[0] + 1 if bos_positions else 0
    return sp.decode(tokens[start:])



def load_prompts_from_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {filepath}")
    return prompts


def save_outputs(results, output_path="generated_text.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Mini Hindi-GPT — Generated Text (Task 2)\n")
        f.write("=" * 60 + "\n\n")
        for i, (prompt, generated) in enumerate(results, 1):
            f.write(f"[{i}] Prompt: {prompt}\n")
            f.write(f"    Generated: {generated}\n")
            f.write("-" * 60 + "\n\n")
    print(f"Outputs saved to {output_path}")



if __name__ == "__main__":
    DEFAULT_PROMPTS = [
        "मैं",
        "आज मौसम",
        "भारत एक",
        "मुझे लगता है",
        "यह फिल्म"
    ]

    if len(sys.argv) >= 2:
        prompts     = load_prompts_from_file(sys.argv[1])
        output_file = sys.argv[2] if len(sys.argv) >= 3 else "generated_text.txt"
    else:
        print("No prompt file provided. Using default prompts.")
        prompts     = DEFAULT_PROMPTS
        output_file = "generated_text.txt"

    sp    = load_tokenizer()
    model = load_model()

    print(f"\nGenerating {len(prompts)} outputs (50 tokens each)...\n")
    results = []
    for prompt in prompts:
        print(f"  Prompt: {prompt}")
        text = generate(model, sp, prompt, max_new_tokens=50)
        results.append((prompt, text))
        print(f"  Generated: {text}\n")

    save_outputs(results, output_file)