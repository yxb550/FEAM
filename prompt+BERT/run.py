
import os
import sys
import csv
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
import sys
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, BertConfig, AutoModel
from sklearn.metrics import classification_report
from tqdm.auto import tqdm


# -------------------------------
# Setup
# -------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

log_dir = "/mnt/workspace/nlp_sentiment_analysis/prompt+BERT/logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

sys.stdout = Logger(log_filename)


# -------------------------------
# Dataset
# -------------------------------
class PlainDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, sample_size=None, domain_id=None):
        df = pd.read_csv(file_path, dtype={"sentence": str, "sentiment": int})
        df.dropna(subset=["sentence", "sentiment"], inplace=True)
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        self.texts = df["sentence"].tolist()
        self.labels = df["sentiment"].tolist()
        self.domain_ids = [domain_id] * len(self.labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        domain_id = self.domain_ids[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "domain_labels": torch.tensor(domain_id, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

    @staticmethod
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "domain_labels": torch.stack([item["domain_labels"] for item in batch])
        }


# -------------------------------
# Model
# -------------------------------
class SoftPromptPTCAM(nn.Module):
    def __init__(self, config, local_model_path, prompt_length=20, num_prompts=10, freeze_bert=True, class_weights=None, entropy_weight=0.05):
        super().__init__()
        self.bert = AutoModel.from_pretrained(local_model_path, config=config)
        self.config = config
        self.prompt_length = prompt_length
        self.num_prompts = num_prompts
        self.entropy_weight = entropy_weight
        self.class_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights else None

        # Dynamic soft prompt pool
        self.soft_prompt_pool = nn.Parameter(torch.randn(num_prompts, prompt_length, config.hidden_size))
        self.controller_query_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.controller_key_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Emotion modulation
        self.emotion_modulator = nn.Linear(config.hidden_size, config.hidden_size)

        # Multi-scale convolution
        out_channels = 128
        self.conv_k2 = nn.Conv1d(config.hidden_size, out_channels, kernel_size=2, padding=1)
        self.conv_k3 = nn.Conv1d(config.hidden_size, out_channels, kernel_size=3, padding=1)
        self.conv_k4 = nn.Conv1d(config.hidden_size, out_channels, kernel_size=4, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Topic-aware attention
        self.topic_query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.token_proj = nn.Linear(out_channels * 3, config.hidden_size)
        self.final_classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Domain classifier
        self.domain_classifier = nn.Linear(config.hidden_size, 3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None, domain_labels=None):
        # Step 1: Dynamic prompt selection
        input_embeds = self.bert.embeddings(input_ids)

        with torch.no_grad():
            temp_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_vec = temp_outputs.last_hidden_state[:, 0]
            mask = attention_mask.unsqueeze(-1).float()
            mean_vec = (temp_outputs.last_hidden_state * mask).sum(1) / (mask.sum(1) + 1e-12)

        control_input = torch.cat([cls_vec, mean_vec], dim=-1)
        query = self.controller_query_proj(control_input).unsqueeze(1)

        keys = self.controller_key_proj(
            self.soft_prompt_pool.view(self.num_prompts, -1, self.config.hidden_size).mean(1)
        ).unsqueeze(0).repeat(query.size(0), 1, 1)

        attn_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
        probs = torch.softmax(attn_scores, dim=-1)
        selected_prompt = torch.einsum("bp,pkh->bkh", probs, self.soft_prompt_pool)

        # Step 2: Concatenate prompt with input and run through BERT
        inputs_with_prompt = torch.cat([selected_prompt, input_embeds], dim=1)
        prompt_attention = torch.ones(input_embeds.size(0), self.prompt_length).to(attention_mask.device)
        extended_mask = torch.cat([prompt_attention, attention_mask], dim=1)

        outputs = self.bert(inputs_embeds=inputs_with_prompt, attention_mask=extended_mask)
        seq_out = outputs.last_hidden_state

        # Step 3: Emotion-aware modulation
        mod_gate = torch.sigmoid(self.emotion_modulator(seq_out))
        mod_out = seq_out * mod_gate

        # Step 4: Multi-scale convolution
        conv_in = mod_out.transpose(1, 2)  # [B, H, L]
        c2 = self.pool(self.relu(self.conv_k2(conv_in)))
        c3 = self.pool(self.relu(self.conv_k3(conv_in)))
        c4 = self.pool(self.relu(self.conv_k4(conv_in)))
        cat = torch.cat([c2, c3, c4], dim=1).squeeze(-1)  # [B, 128*3]

        feat = self.dropout(cat)

        # Step 5: Topic-aware attention
        global_cls = seq_out[:, 0]  # [B, H]
        q = self.topic_query_proj(global_cls).unsqueeze(-1)  # [B, H, 1]
        token_features = self.token_proj(feat).unsqueeze(1)  # [B, 1, H]
        attn_weights = torch.softmax(torch.bmm(token_features, q).squeeze(-1), dim=-1)  # [B, 1]
        Fatt = (token_features.squeeze(1) * attn_weights).squeeze(1)  # [B, H]

        logits = self.final_classifier(Fatt)  # [B, num_labels]
        domain_logits = self.domain_classifier(seq_out[:, 0])  # Domain prediction from CLS

        # Step 6: Loss calculation
        loss = None
        if labels is not None and domain_labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device)) if self.class_weights is not None else nn.CrossEntropyLoss()
            cls_loss = loss_fn(logits, labels)
            domain_loss = nn.CrossEntropyLoss()(domain_logits, domain_labels)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            loss = cls_loss + 0.3 * domain_loss + self.entropy_weight * entropy

        return loss, logits



# -------------------------------
# Load & Prepare
# -------------------------------
local_model_path = "/mnt/workspace/nlp_sentiment_analysis/prompt+BERT/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
config = BertConfig.from_pretrained(local_model_path, num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stage 1: Few-shot pretraining
fewshot_files = ["/mnt/workspace/nlp_sentiment_analysis/data/300data.csv"]
fewshot_dataset = ConcatDataset([PlainDataset(f, tokenizer, domain_id=0) for f in fewshot_files])
fewshot_loader = DataLoader(fewshot_dataset, batch_size=16, shuffle=True, collate_fn=PlainDataset.collate_fn)

model = SoftPromptPTCAM(config, local_model_path, prompt_length=10, num_prompts=5, freeze_bert=True).to(device)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16 * len(fewshot_loader))

for epoch in range(1, 31):
    print(f"[Stage 1] Epoch {epoch}/30")
    model.train()
    total_loss = 0
    for batch in tqdm(fewshot_loader, desc="Few-shot Prompt Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = model(**batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss/len(fewshot_loader):.4f}")

torch.save(model.state_dict(), "/mnt/data/soft_prompt_stage1.ckpt")

# Stage 2: Full training
domain_map = {"laptop": 0, "rest": 1, "twitter": 2}
train_files = [
    "/mnt/workspace/nlp_sentiment_analysis/data/rest16_quad_train_simple_fully_augmented.csv",
]

dev_files = [
    "/mnt/workspace/nlp_sentiment_analysis/data/rest16_quad_dev_simple_augmented_plus.csv",
]
test_files = [
    "/mnt/workspace/nlp_sentiment_analysis/data/Twitter/test_data_from_sample_2000.csv",
]

train_dataset = ConcatDataset([
    PlainDataset(path, tokenizer, domain_id=domain_map[domain])
    for domain, path in train_files.items()
])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=PlainDataset.collate_fn)

dev_dataset = ConcatDataset([
    PlainDataset(path, tokenizer, domain_id=domain_map[domain])
    for domain, path in dev_files.items()
])
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=PlainDataset.collate_fn)

test_dataset = ConcatDataset([
    PlainDataset(path, tokenizer, domain_id=domain_map[domain])
    for domain, path in test_files.items()
])
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=PlainDataset.collate_fn)

all_labels = []
for f in train_files.values():
    df = pd.read_csv(f)
    all_labels += df['sentiment'].tolist()

label_counts = Counter(all_labels)
total = sum(label_counts.values())
class_weights = [total / label_counts[i] for i in range(config.num_labels)]
print(f"Auto computed class weights: {class_weights}")

model = SoftPromptPTCAM(
    config, local_model_path,
    prompt_length=10,
    num_prompts=5,
    freeze_bert=False,
    class_weights=class_weights
).to(device)

model.load_state_dict(torch.load("/mnt/data/soft_prompt_stage1.ckpt"), strict=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16 * len(train_loader))

# Training loop
for epoch in range(1, 31):
    print(f"\n[Stage 2] Epoch {epoch}/30")
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = model(**batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Train Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc=f"[Epoch {epoch}] Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            _, logits = model(**batch)
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_trues.extend(batch["labels"].cpu().numpy())

    print(f"\nClassification Report (Epoch {epoch}):")
    print(classification_report(val_trues, val_preds, digits=4, zero_division=0))

# Final test evaluation
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}
        _, logits = model(**batch)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())

print("Final Test Report:")
print(classification_report(trues, preds, zero_division=0, digits=4))
