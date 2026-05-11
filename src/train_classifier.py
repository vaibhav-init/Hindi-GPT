
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import random

import config_task3 as cfg

from tokenizer import (
    load_corpus_file,
    load_classification_data,
    split_classification,
    ClsDataset,
    load_tokenizer
)

from gpt_model import GPTClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.SEED)


sp = load_tokenizer()


def classification_dataloader(csv_path):
    texts, labels = load_classification_data(csv_path)

    train_X, train_y, val_X, val_y = split_classification(texts, labels)

    train_dataset = ClsDataset(train_X, train_y, sp, cfg.BLOCK_SIZE)
    val_dataset   = ClsDataset(val_X, val_y, sp, cfg.BLOCK_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


def build_classifier():
    model = GPTClassifier(
        vocab_size=cfg.VOCAB_SIZE,
        embed_dim=cfg.N_EMBD,
        block_size=cfg.BLOCK_SIZE,
        num_heads=cfg.N_HEAD,
        num_layers=cfg.N_LAYER,
        num_classes=cfg.NUM_CLASSES
    ).to(cfg.DEVICE)

    return model

model = build_classifier()
try:
    model.backbone.load_state_dict(torch.load("../models/gpt_hindi_best.pt"), strict=False)
    print("Loaded pretrained GPT backbone weights")
except:
    print("No pretrained weights found (training from scratch)")

for param in model.backbone.parameters():
    param.requires_grad = False

print("Backbone GPT weights frozen")

for param in model.backbone.blocks[-1].parameters():
    param.requires_grad = True

for param in model.backbone.blocks[-2].parameters():
    param.requires_grad = True


def train_cls_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")

    for x, y in pbar:
        x = x.to(cfg.DEVICE, non_blocking=True)
        y = y.to(cfg.DEVICE, non_blocking=True)

        optimizer.zero_grad()

        logits, loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        acc = correct / total

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}"
        })

    return total_loss / len(loader), acc


def eval_cls(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(cfg.DEVICE, non_blocking=True)
            y = y.to(cfg.DEVICE, non_blocking=True)

            logits, loss = model(x, y)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total

    return avg_loss, acc


def train_classifier(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Early stopping
    best_val_acc = 0
    patience = 5
    counter = 0

    for epoch in range(cfg.MAX_EPOCHS):
        print(f"\nEpoch {epoch+1}/{cfg.MAX_EPOCHS}")

        train_loss, train_acc = train_cls_epoch(model, train_loader, optimizer)
        val_loss, val_acc = eval_cls(model, val_loader)

        # store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc:  {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), "../models/gpt_classifier_best.pt")
            print("Best model saved")
        else:
            counter += 1

        # Early stopping
        if counter >= patience:
            print("Early stopping triggered")
            break

    # Save final model
    torch.save(model.state_dict(), "../models/gpt_classifier_last.pt")

    print(f"\nFinal Validation Accuracy: {best_val_acc:.4f}")

    # Plot Loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (Task 3)")
    plt.savefig("LossCurveTask3.png")
    plt.show()

    # Plot Accuracy
    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve (Task 3)")
    plt.savefig("AccuracyCurveTask3.png")
    plt.show()

    return train_losses, val_losses, train_accs, val_accs


csv_path = "./text_classification_dataset/train.csv"

train_loader_cls, val_loader_cls = classification_dataloader(csv_path)

train_classifier(model, train_loader_cls, val_loader_cls)


def get_correct_predictions(model, loader, num_samples=5):
    model.eval()
    results = []

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    with torch.no_grad():
        for x, y in loader:
            x = x.to(cfg.DEVICE)
            y = y.to(cfg.DEVICE)

            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)

            for i in range(len(x)):
                if preds[i] == y[i]:
                    tokens = [t for t in x[i].tolist() if t != 0]
                    text   = sp.decode(tokens)
                    label  = int(y[i].item())

                    results.append((text, label, label_map[label]))

    # RANDOMLY SAMPLE
    if len(results) > num_samples:
        results = random.sample(results, num_samples)

    return results


model.load_state_dict(torch.load("../models/gpt_classifier_best.pt"))
model.to(cfg.DEVICE)
model.eval()

print("Loaded best classifier model")


samples = get_correct_predictions(model, val_loader_cls)

with open("../results/correct_predictions.txt", "w", encoding="utf-8") as f:
    for text, label, label_name in samples:          
        f.write(f"Label: {label} ({label_name})\n")
        f.write(f"Text: {text}\n")
        f.write("-" * 60 + "\n\n")                   

print("Saved 5 correct predictions")





