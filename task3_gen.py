import torch
import sys
import os
import sentencepiece as spm
import config_task3 as cfg
from task2_model import GPTClassifier


def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load("model_hindi.model")
    return sp


def load_model():
    model = GPTClassifier(
        vocab_size  = cfg.VOCAB_SIZE,
        embed_dim   = cfg.N_EMBD,
        block_size  = cfg.BLOCK_SIZE,
        num_heads   = cfg.N_HEAD,
        num_layers  = cfg.N_LAYER,
        num_classes = cfg.NUM_CLASSES
    ).to(cfg.DEVICE)

    if not os.path.exists("gpt_classifier_best.pt"):
        raise FileNotFoundError("gpt_classifier_best.pt not found. Train Task 3 first.")

    model.load_state_dict(torch.load("gpt_classifier_best.pt", map_location=cfg.DEVICE))
    model.eval()
    print("Model loaded from gpt_classifier_best.pt")
    return model



def preprocess(sp, text):
    ids = sp.encode(text, out_type=int)

    if len(ids) == 0:
        ids = [1]                                      

    # Truncate to block_size
    ids = ids[:cfg.BLOCK_SIZE]

    # Pad if shorter
    ids = ids + [0] * (cfg.BLOCK_SIZE - len(ids))

    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, T)


LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

def classify(model, sp, text):
    x = preprocess(sp, text).to(cfg.DEVICE)

    with torch.no_grad():
        logits, _ = model(x)
        probs     = torch.softmax(logits, dim=-1)[0]
        pred      = torch.argmax(probs).item()

    return pred, LABEL_MAP[pred], probs.tolist()


def load_texts_from_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(texts)} texts from {filepath}")
    return texts



def save_outputs(results, output_path="classification_output.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        for i, (text, label_id, label_name, probs) in enumerate(results, 1):
            f.write(f"[{i}] Predicted Label: {label_id} ({label_name})\n")
            f.write(f"    Confidence — Negative: {probs[0]:.3f} | "
                    f"Neutral: {probs[1]:.3f} | "
                    f"Positive: {probs[2]:.3f}\n")
            f.write(f"    Text: {text[:300]}{'...' if len(text) > 300 else ''}\n")
            f.write("-" * 60 + "\n\n")
    print(f"Outputs saved to {output_path}")


if __name__ == "__main__":

    DEFAULT_TEXTS = [
        "यह फिल्म बेहतरीन है, अक्षय कुमार ने शानदार अभिनय किया और कहानी बहुत अच्छी थी।",
        "फिल्म बहुत बोरिंग थी, कहानी में कोई दम नहीं था और अभिनय भी ठीक नहीं था।",
        "फिल्म ठीक-ठाक थी, न बहुत अच्छी न बहुत बुरी, एक बार देख सकते हैं।",
        "निर्देशक ने बहुत अच्छा काम किया है, हर दृश्य में भावनाएं महसूस होती हैं।",
        "पैसे और समय दोनों बर्बाद हुए, इस फिल्म को देखने की कोई जरूरत नहीं।"
    ]


    if len(sys.argv) >= 2:
        texts       = load_texts_from_file(sys.argv[1])
        output_file = sys.argv[2] if len(sys.argv) >= 3 else "classification_output.txt"
    else:
        print("No input file provided. Using default texts.")
        texts       = DEFAULT_TEXTS
        output_file = "classification_output.txt"

    sp    = load_tokenizer()
    model = load_model()

    
    print(f"\nClassifying {len(texts)} texts...\n")
    results = []
    for text in texts:
        label_id, label_name, probs = classify(model, sp, text)
        results.append((text, label_id, label_name, probs))
        print(f"  Label: {label_id} ({label_name})")
        print(f"  Confidence — Neg: {probs[0]:.3f} | Neu: {probs[1]:.3f} | Pos: {probs[2]:.3f}")
        print(f"  Text: {text[:80]}...\n")

  
    save_outputs(results, output_file)