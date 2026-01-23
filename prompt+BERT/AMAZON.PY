import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, BertConfig, AutoModel
import random
import numpy as np
from tqdm.auto import tqdm
import os
import torch.nn as nn
from sklearn.metrics import classification_report
import sys
from datetime import datetime
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------------
# 少样本提示数据集定义（单标签）
# -------------------------------
class PromptedDataset(Dataset):
    def __init__(self, data_file, tokenizer, few_shot_prompt, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.few_shot_prompt = few_shot_prompt.strip() + "\n"
        # 读取 CSV, 假设有 header: sentence,sentiment
        df = pd.read_csv(data_file, dtype={"sentence": str, "sentiment": int})
        df.dropna(subset=["sentence", "sentiment"], inplace=True)
        self.texts = df["sentence"].tolist()
        self.labels = df["sentiment"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 构造带提示的输入
        prompt_input = f"{self.few_shot_prompt}输入: \"{text}\"\n输出:"
        encoding = self.tokenizer(
            prompt_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

# -------------------------------
# Few-shot 提示模板（3类：0负向，1中性，2正向）
# -------------------------------
few_shot_prompt = """
示例 1
输入: "I hated the service."
输出: 0

示例 2
输入: "It was neither good nor bad, just fine."
输出: 1

示例 3
输入: "Absolutely fantastic experience!"
输出: 2

示例 4
输入: "The food was okay, nothing special."
输出: 1
"""

# -------------------------------
# 并行卷积 + 注意力的 BERT 模型
# -------------------------------
class BERT_PTCAM(nn.Module):
    def __init__(self, config, local_model_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(local_model_path, config=config)
        self.dropout = nn.Dropout(0.4)
        self.emotion_modulator = nn.Parameter(torch.ones(config.hidden_size))
        out_channels = 128
        self.conv_k2 = nn.Conv1d(config.hidden_size, out_channels, kernel_size=2, padding=1)
        self.conv_k3 = nn.Conv1d(config.hidden_size, out_channels, kernel_size=3, padding=1)
        self.conv_k4 = nn.Conv1d(config.hidden_size, out_channels, kernel_size=4, padding=2)
        self.relu = nn.ReLU()
        self.attention_fc = nn.Linear(out_channels * 3, 1)
        self.classifier = nn.Linear(out_channels * 3, config.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state
        mod_out = seq_out * self.emotion_modulator
        conv_in = mod_out.transpose(1, 2)
        c2 = self.relu(self.conv_k2(conv_in))
        c3 = self.relu(self.conv_k3(conv_in))
        c4 = self.relu(self.conv_k4(conv_in))
        min_len = min(c2.size(2), c3.size(2), c4.size(2))
        c2, c3, c4 = c2[:, :, :min_len], c3[:, :, :min_len], c4[:, :, :min_len]
        cat = torch.cat([c2, c3, c4], dim=1).transpose(1, 2)
        attn_scores = self.attention_fc(cat)
        attn_weights = torch.softmax(attn_scores, dim=1)
        feat = torch.sum(cat * attn_weights, dim=1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits4

# -------------------------------
# 参数与多数据集加载
# -------------------------------
local_model_path = "/mnt/workspace/nlp_sentiment_analysis/prompt+BERT/bert-base-chinese"
num_labels = 3  # 3 类
max_length = 128
batch_size = 8
learning_rate = 1e-5
epoch_num = 16

tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 指定各域的文件列表
train_files = [
    "/mnt/workspace/0.2/new/amazon_train_8000.csv"
]
dev_files = [
    "/mnt/workspace/0.2/new/amazon_dev_1000.csv"
]
test_files = [
    "/mnt/workspace/0.2/new/amazon_test_1000.csv"
]

# 为每个文件创建 PromptedDataset 并合并
train_datasets = [PromptedDataset(f, tokenizer, few_shot_prompt, max_length) for f in train_files]
dev_datasets   = [PromptedDataset(f, tokenizer, few_shot_prompt, max_length) for f in dev_files]
test_datasets  = [PromptedDataset(f, tokenizer, few_shot_prompt, max_length) for f in test_files]

train_dataset = ConcatDataset(train_datasets)
dev_dataset   = ConcatDataset(dev_datasets)
test_dataset  = ConcatDataset(test_datasets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=PromptedDataset.collate_fn)
dev_loader   = DataLoader(dev_dataset,   batch_size=batch_size, shuffle=False, collate_fn=PromptedDataset.collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=PromptedDataset.collate_fn)

# -------------------------------
# 训练与评估流程
# -------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)


log_dir = "/mnt/workspace/nlp_sentiment_analysis/prompt+BERT/噪声"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"ptcam_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass  # 兼容 Python 的 `flush`

# 重定向输出到日志文件
sys.stdout = Logger(log_filename)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = BertConfig.from_pretrained(local_model_path, num_labels=num_labels)
model = BERT_PTCAM(config, local_model_path).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
num_steps = epoch_num * len(train_loader)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

# 训练循环
for epoch in range(1, epoch_num+1):
    print(f"Epoch {epoch}/{epoch_num}")
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = model(**batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
    print(f"Train loss: {total_loss/len(train_loader):.4f}")

    # 验证
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Validating"):
            cpu_labels = batch["labels"].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            _, logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=None)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(cpu_labels)
    print(classification_report(trues, preds, zero_division=0, digits=4))

# 设置 t-SNE 参数
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')

# 提取测试集特征（FEAM 输出）和标签
features, labels = [], []
model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Extracting Features for t-SNE"):
        label_batch = batch["labels"].cpu()
        labels.extend([int(x) for x in label_batch.view(-1)])

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.bert(batch["input_ids"], attention_mask=batch["attention_mask"])
        seq_out = outputs.last_hidden_state
        mod_out = seq_out * model.emotion_modulator
        conv_in = mod_out.transpose(1, 2)
        c2 = model.relu(model.conv_k2(conv_in))
        c3 = model.relu(model.conv_k3(conv_in))
        c4 = model.relu(model.conv_k4(conv_in))
        min_len = min(c2.size(2), c3.size(2), c4.size(2))
        c2, c3, c4 = c2[:, :, :min_len], c3[:, :, :min_len], c4[:, :, :min_len]
        cat = torch.cat([c2, c3, c4], dim=1).transpose(1, 2)
        attn_scores = model.attention_fc(cat)
        attn_weights = torch.softmax(attn_scores, dim=1)
        feat = torch.sum(cat * attn_weights, dim=1)
        features.append(feat.cpu())

# 合并特征与标签
features = torch.cat(features).numpy()
labels = np.array(labels)

# t-SNE 降维
tsne_results = tsne.fit_transform(features)

# 可视化
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=tsne_results[:, 0], y=tsne_results[:, 1],
    hue=labels,
    palette=sns.color_palette("hsv", n_colors=3),
    alpha=0.7,
    s=50
)
plt.title("t-SNE of FEAM Features (Amazon Reviews)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.legend(title="Sentiment Class", labels=["Negative (0)", "Neutral (1)", "Positive (2)"])
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "tsne_visualization.png"))  # 保存图像
plt.show()  # 显示图像


# 测试
print("\nFinal Test Performance:")
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        cpu_labels = batch["labels"].numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        _, logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=None)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        trues.extend(cpu_labels)
print(classification_report(trues, preds, zero_division=0, digits=4))
