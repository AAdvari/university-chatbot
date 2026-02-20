import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dotenv
dotenv.load_dotenv()
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, classification_report



MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"
TRAIN_FILE = "./data/train.json"
TEST_FILE = "./data/test.json"

BATCH_SIZE = 16
EPOCHS = 7
LR = 2e-5
MAX_LENGTH = 128

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


def load_data(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["question"] for item in data]
    labels = [int(item["is_ambiguous"]) for item in data]

    return texts, labels


train_texts, train_labels = load_data(TRAIN_FILE)
test_texts, test_labels = load_data(TEST_FILE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TextDataset(Dataset):

    def __init__(self, texts, labels):

        self.texts = texts
        self.labels = labels

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


train_dataset = TextDataset(train_texts, train_labels)
test_dataset = TextDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class AmbiguityClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            2
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


model = AmbiguityClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

print("\nTraining...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")


print("\nEvaluating...")

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(all_labels, all_preds)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(
    all_labels,
    all_preds,
    target_names=["Specific", "Ambiguous"]
))


def predict(text):

    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        logits = model(input_ids, attention_mask)

        pred = torch.argmax(logits, dim=1).item()

    return "AMBIGUOUS" if pred else "Specific"


print("\nSanity check:")

print("شهریه چقدر است؟ →", predict("شهریه چقدر است؟"))

print("شهریه کارشناسی ارشد ۱۴۰۳ چقدر است؟ →",
      predict("شهریه کارشناسی ارشد ۱۴۰۳ چقدر است؟"))

SAVE_DIR = "./saved_ambiguity_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save model weights
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))

# Save tokenizer (very important)
tokenizer.save_pretrained(SAVE_DIR)

# Save config
config = {
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "label_mapping": {
        0: "Specific",
        1: "Ambiguous"
    }
}

with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("Model saved successfully.")