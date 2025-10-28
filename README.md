# English-to-Vietnamese Machine Translation (Transformer & GPT)

This project explores the development of machine translation models, comparing architectures built from scratch (Transformer, GPT) against pre-trained models.

## Project Objective

The primary goal is to build, train, and evaluate sequence-to-sequence models for English-to-Vietnamese translation. The project places a strong emphasis on modern tokenization techniques and comparative model analysis.

## Methodology

### 1. Tokenization

* **Byte Pair Encoding (BPE):** Implemented BPE for subword tokenization, which is highly effective for handling rare words and morphologically rich languages.
* **Multilingual Shared Vocabulary:** Utilized a shared vocabulary for both English and Vietnamese to improve token efficiency and model learning.

### 2. Model Architectures

Several models were built and compared:

* **Transformer (From Scratch):** Implemented the full Transformer architecture (encoder-decoder) from the ground up.
* **GPT (From Scratch):** Adapted the GPT (decoder-only) architecture for sequence-to-sequence tasks.
* **Pre-trained Models:** Leveraged pre-trained Transformer and GPT models as a baseline and for fine-tuning.

### 3. Evaluation

Model performance was assessed using standard machine translation metrics:

* BLEU Score
* ROUGE Score

## Technologies Used

* **Language:** Python
* **Core Libraries:** TensorFlow / Keras (or PyTorch)
* **NLP/Transformers:** transformers (Hugging Face)
* **Tools:** Jupyter Notebook, Google Colab

## How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/thCharlotte/Machine-Translation-Model.git](https://github.com/thCharlotte/Machine-Translation-Model.git)
    ```
2.  Install the required dependencies (it's recommended to create a requirements.txt file):
    ```bash
    pip install tensorflow pandas transformers
    ```
3.  Open and run the Jupyter Notebooks (.ipynb) to see the implementation of BPE, model training (from scratch and pretrained), and evaluation.

## Notebooks

* `BPE_tokenizer.ipynb`: Implements the Byte Pair Encoding algorithm.
* `transformer_fs_v5.ipynb`: Transformer model built from scratch.
* `gpt-fs-ver2.ipynb`: GPT model built from scratch.
* `transformer-pretrained-v1.ipynb`: Fine-tuning a pre-trained Transformer.
* `pretrained_gpt_v3.ipynb`: Fine-tuning a pre-trained GPT model.
