import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
import warnings
import dotenv
dotenv.load_dotenv()
# Silence HF logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import logging
logging.set_verbosity_error()

# Silence warnings
warnings.filterwarnings("ignore")
class AmbiguityClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
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


class AmbiguityDetector:
    def __init__(self, model_dir, device=None):

        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.device = device

        # Load config
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        self.max_length = config["max_length"]
        self.label_mapping = config["label_mapping"]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load model
        self.model = AmbiguityClassifier(config["model_name"])
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
        )

        self.model.to(device)
        self.model.eval()

    def predict(self, text):
        if text is None or not isinstance(text, str):
            text = str(text) if text is not None else ""
        # Normalize to plain str and ensure UTF-8; pass as list for tokenizer API
        text = text.strip().encode("utf-8", errors="replace").decode("utf-8")
        if not text:
            return self.label_mapping["0"]  # default to Specific when empty

        encoding = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).item()

        return self.label_mapping[str(pred)] if isinstance(self.label_mapping, dict) else self.label_mapping[pred]