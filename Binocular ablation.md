Chat link: [https://chatgpt.com/share/6928d625-c29c-8011-89e5-bce4bd8884b8](https://chatgpt.com/share/6928d625-c29c-8011-89e5-bce4bd8884b8) 

https://github.com/ahans30/Binoculars

Nice, this is exactly the right next step: **“Binoculars vs supervised detectors”** is a super compelling story.

I’ll give you a *very* concrete plan focused on:

1. How to structure the dataset for supervised models.

2. How to do clean train/val/test splits without leakage.

3. How to inject the “knobbed” generations (temperature/top-p/decoding/back-translation) as *augmented data* in different training regimes.

I’ll assume you’re using **TuringBench humans** as the core dataset and generating your own machine texts as we discussed.

---

## **0\. Mental model: what you’re actually comparing**

You now have two detection paradigms:

* **Binoculars** \= zero-shot, LM-based score \+ threshold.

* **Supervised classifier** \= binary classifier trained on examples `[text → {human, machine}]`.

Your data pipeline should be built so that:

* **Both systems see the *same* test texts**, so you can compare apples to apples.

* You can explicitly test:

  * In-distribution performance of the classifier (train & test on same kind of machine text).

  * **Robustness to decoding/back-translation**: test on machine texts generated with knobs it *never saw* during training.

So the core idea:

Split at the *prompt level* first, then derive all variants (human, machine for each knob) and make sure all those variants stay in the same split.

---

## **1\. Build a “master table” of prompts and texts**

Start from N human texts (TuringBench).

### **1.1. For each original human sample**

Create:

* `id`: unique identifier (e.g., `"tb_000123"`).

* `human_text`: the original human text (possibly truncated to ≤ 512 tokens).

* `prompt`: first k tokens (e.g., 50\) of `human_text` decoded back to text.

So your base record is:

`{`  
  `"id": "tb_000123",`  
  `"prompt": "First 50 tokens …",`  
  `"human_text": "Full original human text (maybe truncated)"`  
`}`

### **1.2. Generate machine variants per knob**

For each `id` and `prompt`, generate:

* `machine_default`: text generated with your **baseline** decoding config (e.g., T=0.7, top\_p=0.95, nucleus).

* `machine_T0.1_p0.9`, `machine_T1.0_p0.98`, etc.

* `machine_greedy`, `machine_beam4`, `machine_topk50`, `machine_topp0.95`, etc.

* Optionally:

  * `machine_default_BT`: back-translated version of `machine_default`.

  * `human_BT`: back-translated version of `human_text`.

So your full “master table” row might look like:

`{`  
  `"id": "tb_000123",`  
  `"prompt": "...",`  
  `"human_text": "...",`  
  `"human_BT": "...",                     // optional`  
  `"machine_default": "...",`  
  `"machine_T0.1_p0.9": "...",`  
  `"machine_T0.3_p0.9": "...",`  
  `"machine_T1.0_p0.98": "...",`  
  `"machine_greedy": "...",`  
  `"machine_beam4": "...",`  
  `"machine_topk50": "...",`  
  `"machine_default_BT": "..."            // optional`  
`}`

Think of this as the **canonical source** for all supervised and Binoculars experiments.

---

## **2\. Splitting: train / val / test at the prompt level**

This is critical: you **must not** let any text derived from a given `id` leak across splits.

### **2.1. Prompt-level split**

Suppose you’ve got N IDs (say N=1500). Do something like:

* Shuffle the list of IDs once with a fixed seed.

* Split into:

  * **Train IDs**: 60% (e.g., 900\)

  * **Val IDs**: 20% (e.g., 300\)

  * **Test IDs**: 20% (e.g., 300\)

Now:

* All variants for an ID live in the same split:

  * `id` in train → `human_text`, *all* `machine_*`, `*BT` stay in train set.

  * Similarly for val/test.

This prevents the classifier from seeing almost-identical text (original vs BT vs different decoding) in train and then “cheating” on test.

---

## **3\. Turn the master table into supervised datasets**

Now we’ll define several **views** of the data for supervised models.

### **3.1. Base supervised dataset (no augmentation)**

This is your simplest “vanilla” supervised setting:

* **Positive class (label=0)**: human texts.

* **Negative class (label=1)**: machine texts from the **default** decoding config only.

#### **Train set**

For every `id` in `train_ids`:

Add 1 human example:

 `(text = human_text, label = 0)`

* 

Add 1 machine example:

 `(text = machine_default, label = 1)`

* 

You get a perfectly balanced base training set (size \= 2 × |train\_ids|).

#### **Val & Test sets (in-distribution view)**

Same selection:

* For each `id` in val/test:

  * `(human_text, 0)`

  * `(machine_default, 1)`

This defines **in-distribution** evaluation for your supervised classifier: it tests exactly the distribution it was trained on.

You can train:

* A simple **TF-IDF \+ Logistic Regression** baseline.

* A stronger **RoBERTa / DeBERTa** classifier fine-tuned on this dataset.

---

### **3.2. Knobbed test sets (for robustness analysis)**

Now define **additional test sets** that you will NOT use for training the supervised model.

For each `id` in `test_ids`, and for each knobbed generation `machine_T0.1_p0.9`, `machine_T1.0_p0.98`, `machine_greedy`, etc., build separate test splits:

#### **Example: Test\_T0.1\_p0.9**

* Human side:

  * `(human_text, 0)` for all test\_ids.

* Machine side:

  * `(machine_T0.1_p0.9, 1)` for all test\_ids.

This gives you:

* `Test_default` (baseline): human vs machine\_default.

* `Test_T0.1_p0.9`: human vs machine\_T0.1\_p0.9.

* `Test_T1.0_p0.98`: human vs machine\_T1.0\_p0.98.

* `Test_greedy`, `Test_beam4`, `Test_topk50`, etc.

Crucially: **the classifier was trained only on Test\_default-style machines**, so all these other test sets are **distribution shifts** of the negative class.

You’ll evaluate:

* Supervised classifier and Binoculars on each test set.

* Compare how sharply performance drops (or stays stable) as you change knobs.

---

### **3.3. Back-translation test sets**

Now define BT-specific evaluations.

#### **Scenario A: both sides back-translated**

For each `id` in `test_ids`:

* `(human_BT, 0)`

* `(machine_default_BT, 1)`

Call this `Test_BT_both`.

You evaluate:

* Can the classifier still distinguish AI vs human after both have been run through the same translation pipeline?

* Compare with Binoculars on the same set.

#### **Scenario B: attack where only machine is back-translated**

For each `id` in `test_ids`:

* `(human_text, 0)`

* `(machine_default_BT, 1)`

Call this `Test_BT_attack`.

Interpretation:

* This approximates an **adversarial user** who rewrites AI text via back-translation to evade detection.

* You check how often classifier (and Binoculars) misclassify `machine_default_BT` as human.

---

### **3.4. Optional: OOD model test set (original TuringBench machines)**

If you still have the original TuringBench **machine** texts:

For each `id` in `test_ids` where there is a corresponding TuringBench machine sample:

* `(human_text, 0)`

* `(turingbench_machine, 1)`

Call this `Test_TB_orig`.

This tests:

* Zero-shot generalization of your supervised classifier to **different underlying generators**.

* It mirrors one aspect of the Binoculars paper (one detector, multiple generators).

---

## **4\. Augmented training with knobs (for robustness)**

Up to now, we trained the classifier on **only one** machine distribution (`machine_default`) and used knobs purely as unseen distribution shifts.

You can also define **augmented training regimes** and compare:

### **4.1. Training Regime A (baseline)**

As above:

* Train on only:

  * `human_text` (train ids) as label 0\.

  * `machine_default` (train ids) as label 1\.

### **4.2. Training Regime B (multi-knob augmentation)**

Here, you try to make the classifier more robust by adding some machine variants into training.

For each `id` in `train_ids`:

* Always include 1 human: `(human_text, 0)`.

* For machine side, include multiple variants (simple options):

#### **Option B1: full multi-knob**

Add all machine variants:

 `(machine_default, 1)`  
`(machine_T0.1_p0.9, 1)`  
`(machine_T1.0_p0.98, 1)`  
`(machine_greedy, 1)`  
`(machine_beam4, 1)`  
`(machine_topk50, 1)`  
`...`

* 

You now have a **many-to-one** mapping:

* 1 human per id, K machine variants per id.

If you want balance, you can:

* Either **up-sample** humans (repeat them),

* Or randomly **subsample machines** so that total human and machine counts are similar.

#### **Option B2: sampled multi-knob**

To keep training size manageable:

* For each `id`, randomly pick **one** machine variant among a small set of knobs for training (e.g., default, T0.3\_p0.9, greedy).

This gives each id maybe 1–2 machine examples and avoids blowing up dataset size.

### **4.3. Training Regime C (BT-augmented)**

To specifically improve robustness to back-translation:

For each `id` in `train_ids`:

* For humans:

  * `(human_text, 0)`

  * Optionally also `(human_BT, 0)` as a “style noise” positive sample.

* For machines:

  * `(machine_default, 1)`

  * `(machine_default_BT, 1)`

This trains the classifier to:

* Recognize machine texts both **before** and **after** BT.

* See humans in both raw and BT style.

### **4.4. Evaluation under different regimes**

For **each training regime** (A, B, C):

* Train your classifier.

* Evaluate on the *same* battery of test sets:

  * `Test_default`

  * `Test_T0.1_p0.9`

  * `Test_T1.0_p0.98`

  * `Test_greedy`, `Test_beam4`, etc.

  * `Test_BT_both`

  * `Test_BT_attack`

  * Optional `Test_TB_orig`

Then you can show:

* How augmentation changes **in-distribution** performance.

* How it impacts **robustness** (performance on unseen knobs, BT, new models).

* Compare that pattern to Binoculars, which is not fine-tuned at all.

---

## **5\. Practical data representation**

### **5.1. Long-form “flattened” supervised dataset**

Once you’ve defined which entries go into which split \+ regime, you’ll want a simple table for training code:

Columns:

* `id`

* `split` (train / val / test)

* `regime` (A/B/C or just mark in code)

* `source` (human / machine\_default / machine\_T0.1\_p0.9 / machine\_default\_BT / etc.)

* `text`

* `label` (0=human, 1=machine)

Example row:

`{`  
  `"id": "tb_000123",`  
  `"split": "train",`  
  `"source": "machine_T0.3_p0.9",`  
  `"text": "Generated continuation ...",`  
  `"label": 1`  
`}`

Your training script just filters on:

* `split == "train"` for training.

* `split == "val"` for tuning.

* Then for evaluation, filter `split == "test"` and `source` patterns to construct each test scenario.

### **5.2. Balancing and sampling**

For each supervised *training* dataset:

* Ensure you’re not massively skewed toward negative class from all the machine variants.

* Typical choices:

  * Maintain \~50/50 or at least 40/60 human:machine ratio.

  * If you add a lot of machine variants per id, **downsample machines** or **upsample humans**.

This keeps training stable and makes results easier to interpret.

---

## **6\. Putting it together with Binoculars**

Important: the **splits & test sets** you just designed can be reused **verbatim** for Binoculars.

For each test set you constructed (e.g., `Test_T0.1_p0.9`):

* Run Binoculars on `text` of:

  * human rows (label=0),

  * machine rows (label=1),

* Compute:

  * ROC-AUC

  * TPR at fixed low FPR (e.g., 0.1% or 0.01%)

  * Precision, recall, etc.

Then, for each test set you have a table like:

| Test Set | Model | ROC-AUC | TPR @ 0.01% FPR | F1 |
| ----- | ----- | ----- | ----- | ----- |
| Test\_default | Supervised A | 0.99 | 0.95 | ... |
|  | Supervised B | 0.99 | 0.96 | ... |
|  | Binoculars | 0.98 | 0.93 | ... |
| Test\_T1.0\_p0.98 | Supervised A | 0.90 | 0.40 | ... |
|  | Supervised B | 0.95 | 0.70 | ... |
|  | Binoculars | 0.96 | 0.85 | ... |
| Test\_BT\_attack | Supervised A | 0.80 | 0.30 | ... |
|  | Supervised C | 0.90 | 0.60 | ... |
|  | Binoculars | 0.92 | 0.70 | ... |

This becomes the heart of your results section.

---

## **7\. Summary of the plan in one breath**

* Build a **prompt-level master table**: for each human text, store prompt, human text, and all machine variants (different decoding knobs, plus BT versions).

* Split IDs into **train/val/test**, and keep all variants of each ID in the same split.

* For supervised:

  * **Base training**: train on human vs machine\_default (Regime A).

  * **Augmented training**: add machine variants (Regime B), and/or back-translated human/machine (Regime C).

* For evaluation:

  * Construct multiple **test scenarios**: default, each knob, BT-both, BT-attack, OOD (TuringBench original machines).

  * Evaluate **both** supervised classifiers and Binoculars on exactly the same test sets.

If you want, next I can help you:

* Design exact **supervised model architectures** (e.g., TF-IDF+LR and a small RoBERTa) and how much training data you need.

* Or draft your **“Dataset & Experimental Setup”** section in polished LaTeX with this split/augmentation story clearly spelled out.

