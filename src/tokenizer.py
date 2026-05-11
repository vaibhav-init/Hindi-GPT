import os
import random
import unicodedata
import pandas as pd
import torch
from torch.utils.data import Dataset
import sentencepiece as spm


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def load_corpus_file(corpus_path: str):
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        texts = [normalize(line.strip()) for line in f if line.strip()]
    return texts


def load_corpus_folder(corpus_path: str):
    texts = []
    for fname in os.listdir(corpus_path):
        if fname.endswith('.txt'):
            with open(os.path.join(corpus_path, fname), 'r', encoding='utf-8', errors='ignore') as f:
                texts.extend([normalize(line.strip()) for line in f if line.strip()])
    return texts


def split_corpus(texts, ratio=0.9):
    random.shuffle(texts)
    cut = int(len(texts) * ratio)
    return texts[:cut], texts[cut:]


def train_tokenizer(train_texts):
    tmp_file = "spm_input.txt"
    with open(tmp_file, "w", encoding="utf-8") as f:
        for t in train_texts:
            f.write(t + "\n")

    spm.SentencePieceTrainer.train(
        input=tmp_file,
        model_prefix="../models/tokenizer/model_hindi",
        vocab_size=5000,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<PAD>",
        unk_piece="<UNK>",
        bos_piece="<BOS>",
        eos_piece="<EOS>",
        character_coverage=0.9995,
        num_threads=4,
    )
    os.remove(tmp_file)
    print("Tokenizer trained and saved")


def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load("../models/tokenizer/model_hindi.model")
    print("Model Loaded!")
    return sp


def encode(sp, text: str):
    return sp.encode(text, out_type=int)


def decode(sp, ids):
    return sp.decode(ids)


class LMDataset(Dataset):
    def __init__(self, texts, sp, block_size, stride=128):   
        self.block_size = block_size
        self.samples = []
        for text in texts:
            ids = [sp.piece_to_id("<BOS>")] + encode(sp, text) + [sp.piece_to_id("<EOS>")]

            if len(ids) <= block_size:
                # Short sequence — single sample, pad if needed
                chunk = ids + [0] * (block_size + 1 - len(ids))
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:],  dtype=torch.long)
                self.samples.append((x, y))
            else:
                # Sliding window with stride    
                for start in range(0, len(ids) - block_size, stride):
                    chunk = ids[start: start + block_size + 1]
                    if len(chunk) < block_size + 1:
                        chunk = chunk + [0] * (block_size + 1 - len(chunk))
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:],  dtype=torch.long)
                    self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_classification_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    text_col  = [c for c in df.columns if "text"  in c][0]
    label_col = [c for c in df.columns if c != text_col][0]
    texts  = df[text_col].astype(str).map(normalize).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


def split_classification(texts, labels):
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    cut = int(len(combined) * 0.9)
    train, val = combined[:cut], combined[cut:]
    return (
        [x for x, _ in train], [y for _, y in train],
        [x for x, _ in val], [y for _, y in val],
    )


class ClsDataset(Dataset):
    def __init__(self, texts, labels, sp, block_size):
        self.block_size = block_size
        self.samples = []

        for text, label in zip(texts, labels):
            ids = encode(sp, str(text))

            if len(ids) == 0:
                ids = [1]  # <UNK>

            # TAKE LAST TOKENS INSTEAD OF FIRST
            if len(ids) > block_size:
                ids = ids[-block_size:]  

            # Pad if shorter
            pad_len = block_size - len(ids)
            ids = ids + [0] * pad_len

            x = torch.tensor(ids, dtype=torch.long)
            y = torch.tensor(label, dtype=torch.long)

            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]