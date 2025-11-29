Robust AI-Text Detection under Decoding Shifts and Paraphrase Attacks

This repository contains the full experimental pipeline for evaluating Binoculars (zero-shot LM-based detector) and supervised detectors (TF-IDF + Logistic Regression, CNN, RoBERTa) on the TuringBench dataset, including temperature-shift, top-p decoding, paraphrase/backtranslation, and held-out generator robustness experiments.

The project is structured to be reproducible, modular, and suitable for academic reporting.

⸻

🚀 Project Overview

Goal

To study the robustness of AI text detectors under distributional shifts such as:
	•	Temperature-based decoding variations
	•	Top-p sampling variations
	•	Paraphrase attacks (back-translation)
	•	Held-out generators unseen in training

We compare:

Zero-shot detector:
	•	Binoculars (Observer–Performer cross-perplexity method)

Supervised detectors:
	•	TF-IDF + Logistic Regression
	•	CNN (Kim CNN model)
	•	RoBERTa-Base fine-tuned

⸻

🧰 Key Features
	•	Loads TuringBench directly from HuggingFace (no JSONL input required)
	•	Generates temperature & top-p variants using rewriting prompts
	•	Performs back-translation using MarianMT
	•	Implements training/eval pipelines for TF-IDF, CNN, RoBERTa
	•	Implements Binoculars scoring and token-level metrics
	•	Produces complete evaluation metrics:
	•	Accuracy, AUROC, AUPRC
	•	TPR@0.01%FPR
	•	Calibration (ECE + reliability plots)
	•	Cross-Perplexity, Cross-KL, JS Divergence (LM models only)
	•	Ablation experiments for both Binoculars and supervised detectors
	•	Evolution heatmaps summarizing robustness degradation

⸻

📁 Repository Structure

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
│   ├── generated/
│   │   ├── temperature/
│   │   ├── topp/
│   │   ├── paraphrase/
│   │   └── heldout/
│   └── predictions/

├── notebooks/
├── reports/
│   ├── figs/
│   └── tables/
├── scripts/
│   ├── load_turingbench.py
│   ├── create_splits.py
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


⸻

📦 Installation

Clone the repo:

git clone <your-repo-url>
cd project_root

Create environment (recommended):

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


⸻

📥 Dataset: TuringBench

We load TuringBench using HuggingFace:

from datasets import load_dataset
ds = load_dataset("turingbench", split="train")

Splits (train/dev/test) are stored as index lists in data/splits/.

⸻

🔧 How to Run the Pipeline

1. Create dataset splits

python scripts/create_splits.py

2. Generate OOD variants

Temperature-shift:

python scripts/generate_variants.py --mode temperature

Top-p shift:

python scripts/generate_variants.py --mode topp

Paraphrase:

python scripts/backtranslate.py

3. Train supervised models

TF-IDF + LR:

python scripts/train_tfidf.py

CNN:

python scripts/train_cnn.py

RoBERTa:

python scripts/train_roberta.py

4. Score using Binoculars

python scripts/score_binoculars.py

5. Evaluate all models on all test sets

python scripts/evaluate_all.py

Outputs are saved in:

reports/tables/
reports/figs/


⸻

🧪 Evaluation Metrics

All models (supervised + Binoculars):
	•	Accuracy
	•	AUROC
	•	AUPRC
	•	TPR at fixed low FPR (0.01%, 0.1%)
	•	FPR at fixed TPR
	•	Calibration (ECE + reliability diagram)

Only LM-based methods (Binoculars / Causal LMs):
	•	Perplexity
	•	Cross-Perplexity
	•	Cross-KL Divergence
	•	Jensen–Shannon Divergence
	•	Token-rank statistics
	•	Surprisal curves
	•	Binoculars score = PPL / Cross-PPL

⸻

🔬 Ablation Studies

Binoculars Ablations
	•	Observer/Performer model pairs
	•	max_token_observed
	•	Score formula variants
	•	Truncation strategy variations
	•	Thresholding strategies

Supervised Ablations
	•	Training set size
	•	Removing generator classes (held-out generator)
	•	TF-IDF: word vs char vs hybrid
	•	CNN: embedding variants
	•	RoBERTa: seq length 128/256/512
	•	Augmented vs non-augmented training

All ablations are configured via configs/ablation_config.yaml.

⸻

📊 Outputs & Reports

You will obtain:
	•	ROC and PR curves
	•	Low-FPR zoomed ROC
	•	Calibration curves
	•	JS divergence & cross-KL distributions
	•	Evolution matrix heatmaps
	•	Tables of all metrics across all model × dataset-variant combinations
	•	Final PDF report in reports/final_report.pdf

⸻

🧵 Reproducibility
	•	All random seeds fixed
	•	All splits saved as JSON
	•	All generated variants stored on disk
	•	Experiment configs stored in configs/*.yaml
	•	Results logged as JSONL & CSV

⸻

👥 Contributors

Add your team names here.

⸻

📜 License

Specify license here (MIT recommended).
