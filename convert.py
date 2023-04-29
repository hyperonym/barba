"""
Use this script to convert a model to the SavedModel format.
"""
import sys
import tempfile

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

if len(sys.argv) < 3:
    sys.exit("usage: python convert.py REPO_ID EXPORT_DIR")

with tempfile.TemporaryDirectory() as cache_dir:
    model = TFAutoModelForSequenceClassification.from_pretrained(
        sys.argv[1], cache_dir=cache_dir
    )
    tf.saved_model.save(model, sys.argv[2])
