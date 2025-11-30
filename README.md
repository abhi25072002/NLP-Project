# Robust AI-Text Detection under Decoding Shifts and Paraphrase Attacks

This repository contains the full experimental pipeline for evaluating Binoculars (zero-shot LM-based detector) and supervised detectors (TF-IDF + Logistic Regression, CNN, RoBERTa) on the TuringBench dataset, including temperature-shift, top-p decoding, paraphrase/backtranslation, and held-out generator robustness experiments.

The project is structured to be reproducible, modular, and suitable for academic reporting.

---

## 🚀 Project Overview

### Goal

To study the robustness of AI text detectors under distributional shifts such as:
- Temperature-based decoding variations
- Top-p sampling variations
- Paraphrase attacks (back-translation)
- Held-out generators unseen in training

### We compare:

**Zero-shot detector:**
- Binoculars (Observer–Performer cross-perplexity method)

**Supervised detectors:**
- TF-IDF + Logistic Regression
- CNN (Kim CNN model)
- RoBERTa-Base fine-tuned

---

## 🧰 Key Features
- Loads TuringBench directly from HuggingFace (no JSONL input required)
- Generates temperature & top-p variants using rewriting prompts
- Performs back-translation using MarianMT
- Implements training/eval pipelines for TF-IDF, CNN, RoBERTa
- Implements Binoculars scoring and token-level metrics
- Produces complete evaluation metrics:
    - Accuracy, AUROC, AUPRC
    - TPR@0.01%FPR
    - Calibration (ECE + reliability plots)
    - Cross-Perplexity, Cross-KL, JS Divergence (LM models only)
    - Ablation experiments for both Binoculars and supervised detectors
    - Evolution heatmaps summarizing robustness degradation

---

## 📁 Repository Structure

```
project_root/
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── generation_config.yaml
│   ├── eval_config.yaml
│   └── ablation_config.yaml
├── data/
│   ├── hf_cache/
│   ├── splits/

├── notebooks/
├── reports/
│   ├── figs/
│   └── tables/
├── scripts/
│   ├── load_turingbench.py
│   ├── create_supervised_datasets.py
│   ├── generate_variants.py
│   ├── train_tfidf.py
│   ├── train_cnn.py
│   ├── train_roberta.py
│   ├── score_binoculars.py
│   ├── evaluate_all.py
│   └── backtranslate.py
├── src/
│   ├── data/
│   ├── generation/
│   ├── models/
│   ├── binoculars/
│   ├── eval/
│   ├── utils/
│   └── ablation/
├── requirements.txt
└── README.md
```

---

## 📦 Installation

Clone the repo:

```bash
git clone <your-repo-url>
cd project_root
```

Create environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 🔑 Hugging Face Setup (Required)

To download the Llama-2 model and other gated resources, you must:

1.  **Create a Hugging Face Token**:
    *   Go to [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens).
    *   Click **"Create new token"**.
    *   Select **"Read"** (or "Fine-grained" with Read permissions).
    *   Copy this token; you will need it for the environment setup below.

2.  **Request Access to Llama-2**:
    *   Visit the [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model page.
    *   Accept the license agreement and request access.
    *   Wait for the approval email (usually fast).

### ⚠️ Critical Environment Setup (PACE/HPC)

To avoid **Disk Quota Exceeded** errors and ensure access to gated models (Llama-2), run these commands **before** running any scripts:

```bash
# 1. Set Cache to Scratch (Replace with your actual scratch path)
export HF_HOME=/home/hice1/ajd6/scratch/NLP-Project/hf/hf_home
export TRANSFORMERS_CACHE=/home/hice1/ajd6/scratch/NLP-Project/hf/hf_cache
export HF_DATASETS_CACHE=/home/hice1/ajd6/scratch/NLP-Project/hf/data_cache
export XDG_CACHE_HOME=/home/hice1/ajd6/scratch/NLP-Project/hf/hf_cache
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $XDG_CACHE_HOME

# 2. Authenticate with Hugging Face (Required for Llama-2)
export HF_TOKEN="your_hf_token_here"
# OR run: huggingface-cli login
```

---

## ⚙️ Configuration

### Calibrating the "Knobs"

You can customize all generation parameters in **`configs/generation_config.yaml`**.

**Key settings to tweak:**
- **`default`**: The baseline settings for `machine_default` (e.g., `temperature: 0.7`).
- **`variants.temperature.values`**: List of temperatures to test (e.g., `[0.1, 1.0, 1.5]`).
- **`variants.top_p.values`**: List of top-p values (e.g., `[0.9, 0.99]`).
- **`generator_model`**: Change the model used for generation (e.g., `gpt2-xl`, `opt-1.3b`).

---

## 📥 Dataset: TuringBench
Due to the Hugging Face dataset script being deprecated, we use a local copy of the dataset.

1.  **Download** `turingbench.zip` (e.g., from Canvas or shared drive).
2.  **Upload** it to the `data/` directory.
3.  **Prepare** the data:

```bash
python scripts/prepare_local_turingbench.py
```

This extracts the data and consolidates the human texts into `data/human_turingbench.csv`.

---

## 🔧 How to Run the Pipeline

### 1. Create Base Dataset (Human + Machine Default)

Run the following command to load TuringBench, create the master table, and generate the baseline machine text (`machine_default`):

```bash
python scripts/generate_default_dataset.py
```

**What this does:**
- Loads human texts from TuringBench.
- Extracts prompts.
- Generates `machine_default` text using settings in `configs/generation_config.yaml`.
- Saves everything to `data/master_table.jsonl`.

### 2. Generate Robustness Variants (OOD)

To test robustness, generate variants with different "knobs" (temperature, top-p, etc.):

```bash
python scripts/generate_variants.py --mode all
```

### 3. Generate Back-Translations (Paraphrase Attack)

Run back-translation to create `human_BT` and `machine_default_BT` variants:

```bash
python scripts/backtranslate.py
```

### 4. Create Supervised Splits & Test Sets

Finally, create the actual train/val/test files for supervised learning:

```bash
python scripts/create_supervised_datasets.py
```

**This creates:**
- **Base Datasets** (`data/supervised/base/`): Standard Human vs Machine Default.
- **Knobbed Test Sets** (`data/supervised/knobs/`): Test sets for each OOD variant (e.g., `test_T0.1_p0.9.jsonl`).
- **Back-Translation Sets** (`data/supervised/backtrans/`): Test sets for paraphrase attacks.

### 5. Train supervised models

**TF-IDF + LR:**

```bash
python scripts/train_tfidf.py
```

**CNN:**

```bash
python scripts/train_cnn.py
```

**RoBERTa:**

```bash
python scripts/train_roberta.py
```

### 6. Score using Binoculars

```bash
python scripts/score_binoculars.py
```

### 7. Evaluate all models on all test sets

```bash
python scripts/evaluate_all.py
```

Outputs are saved in:
- `reports/tables/`
- `reports/figs/`

---

## 🧪 Evaluation Metrics

**All models (supervised + Binoculars):**
- Accuracy
- AUROC
- AUPRC
- TPR at fixed low FPR (0.01%, 0.1%)
- FPR at fixed TPR
- Calibration (ECE + reliability diagram)

**Only LM-based methods (Binoculars / Causal LMs):**
- Perplexity
- Cross-Perplexity
- Cross-KL Divergence
- Jensen–Shannon Divergence
- Token-rank statistics
- Surprisal curves
- Binoculars score = PPL / Cross-PPL

---

## 🔬 Ablation Studies

**Binoculars Ablations**
- Observer/Performer model pairs
- max_token_observed
- Score formula variants
- Truncation strategy variations
- Thresholding strategies

**Supervised Ablations**
- Training set size
- Removing generator classes (held-out generator)
- TF-IDF: word vs char vs hybrid
- CNN: embedding variants
- RoBERTa: seq length 128/256/512
- Augmented vs non-augmented training

All ablations are configured via `configs/ablation_config.yaml`.

---

## 📊 Outputs & Reports

You will obtain:
- ROC and PR curves
- Low-FPR zoomed ROC
- Calibration curves
- JS divergence & cross-KL distributions
- Evolution matrix heatmaps
- Tables of all metrics across all model × dataset-variant combinations
- Final PDF report in `reports/final_report.pdf`

---

## 🧵 Reproducibility
- All random seeds fixed
- All splits saved as JSON
- All generated variants stored on disk
- Experiment configs stored in `configs/*.yaml`
- Results logged as JSONL & CSV

---

## 👥 Contributors

Add your team names here.

---

## 📜 License

Specify license here (MIT recommended).
