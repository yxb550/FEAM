# FEAM: Feature Enhanced Attention Model

**FEAM** is a robust deep learning framework for sentiment analysis that enhances pre-trained BERT architectures with multi-scale feature extraction and dynamic adaptation. It is specifically designed to perform across diverse domains, from financial reports to social media posts.

---

## 🚀 Key Features

* **Multi-Scale Convolutional Fusion:** Captures local semantic patterns using parallel CNN layers (kernels 2, 3, 4) to complement BERT's global attention.
* **Emotion-Aware Modulation:** Refines hidden states through a learnable modulator to emphasize sentiment-bearing tokens.
* **Dynamic Soft Prompting:** Uses a Query-Key mechanism to select domain-specific prompts, improving performance in few-shot scenarios.
* **Cross-Domain Robustness:** Built-in support for multi-domain training and domain-invariant feature learning.

---

## 📂 Dataset Overview

The repository includes curated datasets across five distinct domains. Each domain folder contains standardized `train`, `test`, and `dev` (validation) splits.

### Supported Domains

* **Amazon:** General product reviews and consumer feedback.
* **Finance:** Financial news, earnings reports, and market sentiment.
* **Laptop:** Technical reviews focusing on hardware and performance.
* **Rest (Restaurant):** Service-industry reviews (food quality, ambiance, etc.).
* **Twitter:** Short-form social media posts with high informal language density.

### Data Format

All datasets are formatted for **3-Class Classification**:

* **Class 0:** Negative
* **Class 1:** Neutral
* **Class 2:** Positive

---

## 🏗️ Model Architecture

The FEAM architecture processes input through several specialized stages:

1. **Contextual Encoding:** BERT-base extracts high-level semantic embeddings.
2. **Feature Modulation:** An `emotion_modulator` gate adjusts token weights:


3. **Parallel Convolution:** Concurrent  operations capture multi-gram features.
4. **Attention Fusion:** Features are aggregated using a topic-aware attention layer before the final classification head.

---

## 🛠️ Getting Started

### 1. Prerequisites

```bash
pip install torch transformers pandas scikit-learn seaborn matplotlib tqdm

```

### 2. Training the Model

To train the model on a specific domain (e.g., Amazon) using the hard-prompt configuration:

```bash
python AMAZON.PY

```

For advanced two-stage training (Soft-Prompt pre-training + full fine-tuning):

```bash
python run.py

```

### 3. Feature Visualization

Generate t-SNE clustering plots to analyze the separation of the three sentiment classes:

```bash
python vis.py 

```
