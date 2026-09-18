# Mental Health Text Classification with LLaMA 3.2

**A comparative NLP project using zero-shot inference, QLoRA, and soft prompt tuning for seven-class text classification.**

This project adapts **Meta LLaMA 3.2 1B Instruct** to classify user-written statements into mental-health-related categories. It compares the untuned model with two parameter-efficient fine-tuning (PEFT) approaches, **Quantized Low-Rank Adaptation (QLoRA)** and **Prompt Tuning**, using the *Sentiment Analysis for Mental Health* dataset.

> **Research disclaimer:** These labels describe patterns in a dataset, not clinical diagnoses or reliable assessments of an individual. This experimental model must not be used for diagnosis, suicide-risk triage, or clinical decision-making.

---

## 📊 Results

Results reported in the project presentation, evaluated on a 5,000-sample test set:

| Approach               |   Accuracy | Weighted F1 | Macro F1 |
| ---------------------- | ---------: | ----------: | -------: |
| Untuned LLaMA Baseline |     24.64% |      0.1781 |     0.16 |
| Prompt Tuning          |     61.28% |      0.6181 |     0.60 |
| QLoRA                  | **75.20%** |  **0.7512** | **0.75** |

QLoRA achieved a reported accuracy improvement of **50.56 percentage points** over the baseline.

**Evaluation caveat:** The supplied preprocessing scripts perform oversampling before splitting the dataset, which may introduce duplicate samples across training and evaluation sets. Consequently, these results represent the original experiment and should not be treated as independently verified generalization benchmarks.

---

## 🎯 Project Objectives

* Classify text into seven mental-health-related categories.
* Evaluate the performance of a pretrained LLaMA model without task-specific fine-tuning.
* Implement QLoRA for memory-efficient fine-tuning.
* Implement Prompt Tuning using trainable soft prompt embeddings.
* Compare the different approaches using accuracy, F1-score, and confusion matrices.
* Explore the trade-offs between computational efficiency and classification performance.

---

## 📂 Dataset

**Dataset:** [Sentiment Analysis for Mental Health, Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

The dataset contains user-written statements associated with seven mental-health-related categories.

**Dataset statistics:**

| Property               | Value  |
| ---------------------- | ------ |
| Original samples       | 53,043 |
| Samples after cleaning | 52,681 |
| Number of classes      | 7      |
| Training split         | 80%    |
| Validation split       | 10%    |
| Testing split          | 10%    |

### Classification Categories

1. Normal
2. Depression
3. Suicidal
4. Anxiety
5. Bipolar
6. Stress
7. Personality Disorder

The categories are labels provided by the dataset and should not be interpreted as medically validated diagnoses.

---

## 🔄 Data Preprocessing

The preprocessing pipeline includes:

**1. Data Cleaning**

* Remove missing statements.
* Drop unnecessary index columns.
* Remove HTML tags using BeautifulSoup.
* Remove URLs using regular expressions.
* Normalize whitespace.

**2. Instruction Formatting**

Each sample is converted into an instruction-response format suitable for supervised LLM fine-tuning.

```text
### Instruction:
Classify the mental health status of the following statement.

### Statement:
<input text>

### Response:
<label>
```

**3. Handling Class Imbalance**

A hybrid resampling strategy is applied:

* Undersample majority classes exceeding 10,000 samples.
* Oversample minority classes containing fewer than 5,000 samples.
* Shuffle the resulting dataset using a fixed random seed.

**4. Dataset Splitting**

The resampled dataset is divided into:

* Training: 80%
* Validation: 10%
* Testing: 10%

The experimental pipeline uses `random_state=42`.

**Important:** For a more reliable evaluation, the original data should be split before oversampling, with resampling applied exclusively to the training partition.

---

## 🧠 Model Architecture

### Base Model

[Meta LLaMA 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

LLaMA is a decoder-only Transformer language model.

The pretrained model is adapted to perform multi-class text classification through instruction-based generation.

Three configurations are evaluated:

### 1. Baseline Model

The original pretrained model is evaluated without task-specific fine-tuning.

This provides a reference for measuring the effects of the adaptation methods.

### 2. QLoRA Fine-Tuning

QLoRA combines 4-bit quantization with Low-Rank Adaptation to reduce the memory required for LLM fine-tuning.

**Quantization Configuration**

| Parameter           | Value   |
| ------------------- | ------- |
| Quantization        | 4-bit   |
| Quantization type   | NF4     |
| Compute dtype       | float16 |
| Double quantization | Enabled |

**LoRA Configuration**

| Parameter      | Value                          |
| -------------- | ------------------------------ |
| Rank           | 16                             |
| Alpha          | 32                             |
| Dropout        | 0.05                           |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Task type      | CAUSAL_LM                      |

The pretrained model weights remain frozen while the additional LoRA adapter parameters are optimized.

The report identifies approximately 13.6 million trainable parameters for this configuration.

### 3. Prompt Tuning

Prompt Tuning adapts the model using a small number of trainable virtual tokens.

Instead of modifying the pretrained model weights, the method learns continuous prompt embeddings that guide the model toward the target classification task.

**Configuration**

| Parameter            | Value     |
| -------------------- | --------- |
| Virtual tokens       | 20        |
| Initialization       | Text      |
| Task type            | CAUSAL_LM |
| Trainable parameters | 40,960    |

The base model remains frozen during training.

QLoRA and Prompt Tuning are evaluated as separate experiments rather than combined into a single model.

---

## ⚙️ Training Configuration

| Parameter               | Value             |
| ----------------------- | ----------------- |
| Epochs                  | 1                 |
| Batch size              | 4                 |
| Gradient accumulation   | 4                 |
| Learning rate           | 2e-4              |
| Maximum sequence length | 160 tokens        |
| QLoRA optimizer         | paged_adamw_32bit |
| QLoRA training hardware | Kaggle P100 GPU   |

The project report records a QLoRA training time of approximately **2 hours and 14 minutes**.

---

## 🛠️ Technologies Used

**Programming Language**

* Python

**Deep Learning and NLP**

* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* TRL
* BitsAndBytes
* Accelerate

**Data Processing**

* Pandas
* NumPy
* Scikit-learn
* BeautifulSoup

**Evaluation and Visualization**

* Matplotlib
* Seaborn
* Scikit-learn Metrics

**Development Environments**

* Kaggle Notebooks
* Google Colab
* Jupyter Notebook

---


## 🚀 Getting Started

### 1. Requirements

A CUDA-compatible GPU is recommended for running the training experiments.

Access to the LLaMA 3.2 model on Hugging Face is required.

### 2. Install Dependencies

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate huggingface_hub pandas numpy scikit-learn beautifulsoup4 matplotlib seaborn
```

### 3. Download the Dataset

Download the dataset from Kaggle:

[Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

Make the `Combined Data.csv` file available to the training environment.

Update the dataset path in the scripts if necessary.

### 4. Hugging Face Authentication

Authenticate through the Hugging Face CLI:

```bash
hf auth login
```

You must have access to the model repository.

Never hard-code authentication tokens into publicly shared scripts.

### 5. Run the Experiments

**QLoRA**

Open `NLP - QLoRA.py` in a notebook-compatible environment and execute the cells sequentially.

The file was exported from a notebook and includes notebook-specific commands, so it may require adjustment before running as a standard Python script.

**Prompt Tuning**

Open `nlp-prompttuning.ipynb` in Jupyter Notebook or Kaggle and execute the cells.

Library versions are not pinned in the original scripts, so compatibility adjustments may be necessary for newer versions of Transformers and TRL.

---

## 📈 Evaluation

The models are evaluated using:

* Accuracy
* Weighted F1-score
* Macro F1-score
* Precision
* Recall
* Classification reports
* Confusion matrices

### Baseline

The pretrained model achieved:

**Accuracy: 24.64%**

**Weighted F1: 0.1781**

The baseline predictions were heavily concentrated in the Normal category.

### Prompt Tuning

After optimizing the virtual prompt tokens, the model achieved:

**Accuracy: 61.28%**

**Weighted F1: 0.6181**

The experiment demonstrated that task-specific prompt embeddings can improve classification performance while training only a small number of parameters.

### QLoRA

The QLoRA model achieved:

**Accuracy: 75.20%**

**Weighted F1: 0.7512**

QLoRA produced the highest reported accuracy and weighted F1-score among the three configurations in this experiment.

---

## 📚 References

1. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
2. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
3. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
4. [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
5. [Sentiment Analysis for Mental Health Dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

---

## Responsible Use

This repository is intended for educational and research purposes only.

The models are not validated medical devices and must not be used to diagnose mental health conditions, assess suicide risk, or replace qualified mental health professionals.

Any further use requires appropriate consideration of data privacy, dataset provenance, consent, fairness, and model reliability.

