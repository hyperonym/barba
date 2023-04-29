"""
Download model and tokenizer from Hugging Face Hub.
"""
import sys

import tensorflow as tf
from huggingface_hub import hf_hub_download
from transformers import TFAutoModelForSequenceClassification

if len(sys.argv) < 3:
    sys.exit("usage: python download.py REPO_ID LOCAL_DIR")

# Download fine-tuned model.
model = TFAutoModelForSequenceClassification.from_pretrained(sys.argv[1])
tf.saved_model.save(model, sys.argv[2])

# Download pretrained tokenizer.
hf_hub_download(
    repo_id="xlm-roberta-large",
    filename="sentencepiece.bpe.model",
    local_dir=sys.argv[2],
    local_dir_use_symlinks=False,
)
