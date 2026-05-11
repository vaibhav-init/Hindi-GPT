
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import config as cfg
from gpt_model import GPTLanguageModel
from tokenizer import (
    train_tokenizer,
    load_corpus_file,
    split_corpus,
    load_tokenizer,
    LMDataset
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)

def prepare_dataloaders(corpus_path):
    texts = load_corpus_file(corpus_path)
    train_texts, val_texts = split_corpus(texts)

    if os.path.exists("../models/tokenizer/model_hindi.model"):
        print("Loading existing tokenizer...")
        sp = load_tokenizer()
    else:
        print("Tokenizer not found. Training now...")
        train_tokenizer(train_texts)
        sp = load_tokenizer()

    train_dataset = LMDataset(train_texts, sp, cfg.BLOCK_SIZE, stride=128) 
    val_dataset   = LMDataset(val_texts,   sp, cfg.BLOCK_SIZE, stride=128)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, sp


def build_model():
    model = GPTLanguageModel(
        vocab_size=cfg.VOCAB_SIZE,
        embed_dim=cfg.N_EMBD,
        block_size=cfg.BLOCK_SIZE,
        num_heads=cfg.N_HEAD,
        num_layers=cfg.N_LAYER
    ).to(cfg.DEVICE)

    return model


def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)

    for x, y in pbar:
        x = x.to(cfg.DEVICE, non_blocking=True)
        y = y.to(cfg.DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(cfg.DEVICE, non_blocking=True)
            y = y.to(cfg.DEVICE, non_blocking=True)
            _, loss = model(x, y)
            total_loss += loss.item()

    avg_loss   = total_loss / len(loader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()


def train(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
    scaler    = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.MAX_EPOCHS
    )

    train_losses = []
    val_losses   = []
    val_ppls     = []

    best_val_loss = float("inf")
    patience = 2
    counter  = 0

    for epoch in range(cfg.MAX_EPOCHS):
        print(f"\nEpoch {epoch+1}/{cfg.MAX_EPOCHS}")

        train_loss          = train_one_epoch(model, train_loader, optimizer, scaler)
        val_loss, val_ppl   = evaluate(model, val_loader)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ppls.append(val_ppl)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val PPL:    {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "../models/gpt_hindi_best.pt")
            print("Model saved (best)")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    torch.save(model.state_dict(), "../models/gpt_hindi_last.pt")

    # print best model perplexity 
    print("\nFinal Evaluation on Best Model")
    model.load_state_dict(torch.load("../models/gpt_hindi_best.pt"))
    best_val_loss, best_val_ppl = evaluate(model, val_loader)
    print(f"Best Model Val Loss:        {best_val_loss:.4f}")
    print(f"Best Model Val Perplexity:  {best_val_ppl:.2f}")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.savefig("TrainingVsValidation.png")
    plt.show()

    return train_losses, val_losses, val_ppls



corpus_path = "./hindi_corpus/combined.txt"

train_loader, val_loader, sp = prepare_dataloaders(corpus_path)
model = build_model()

x, y = next(iter(train_loader))
x = x.to(cfg.DEVICE)
y = y.to(cfg.DEVICE)
logits, loss = model(x, y)
print("Input:",   x.shape)
print("Logits:",  logits.shape)
print("Loss:",    loss.item())


train_losses, val_losses, val_ppls = train(model, train_loader, val_loader)









