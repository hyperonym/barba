"""
Use this script to export a servable Hyperonym Barba model
for end-to-end serving and containerized deployment.
"""
import os
import tempfile

import tensorflow as tf
import tensorflow_text as tf_text
from huggingface_hub import hf_hub_download
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

# Download tokenizer model.
with tempfile.TemporaryDirectory() as cache_dir:
    hf_hub_download(
        repo_id="xlm-roberta-large",
        filename="sentencepiece.bpe.model",
        cache_dir=cache_dir,
        local_dir="models/",
        local_dir_use_symlinks=False,
    )
with open("models/sentencepiece.bpe.model", mode="rb") as file:
    tokenizer_model = file.read()

# Create SentencePiece tokenizer.
tokenizer = tf_text.SentencepieceTokenizer(
    model=tokenizer_model,
    out_type=tf.int32,
    nbest_size=0,
    alpha=1.0,
    reverse=False,
    add_bos=False,
    add_eos=False,
    return_nbest=False,
    name=None,
)

# Load the fine-tuned model.
model = tf.saved_model.load("models/barba")


class Servable(tf.keras.Model):
    """Servable model for end-to-end serving and containerized deployment."""

    def __init__(self):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    def tokenize(self, inputs):
        ids = self.tokenizer.tokenize(inputs) + 1
        ids = tf.where(ids == 1, 3, ids)  # <unk> = 0 -> 3
        return ids

    def concat(self, hypothesis_ids, premise_ids):
        n = hypothesis_ids.bounding_shape()[0]
        bos = tf.zeros([n, 1], dtype=hypothesis_ids.dtype)  # <s> = 0
        eos = tf.ones([n, 1], dtype=hypothesis_ids.dtype) * 2  # </s> = 2
        ids = [bos, hypothesis_ids, eos, eos, premise_ids, eos]
        ids = tf.concat(ids, axis=-1)
        ids = ids.to_tensor(default_value=1)  # <pad> = 1
        return ids[:, :512]

    def infer_ids(self, ids):
        mask = tf.cast(ids != 1, tf.int32)  # <pad> = 1
        fn = self.model.signatures["serving_default"]
        logits = fn(input_ids=ids, attention_mask=mask)["logits"]
        return tf.nn.softmax(logits)[:, 0]

    def infer_text(self, hypothesis, premise):
        hypothesis_ids = self.tokenize(hypothesis)
        premise_ids = self.tokenize(premise)
        ids = self.concat(hypothesis_ids, premise_ids)
        return self.infer_ids(ids)

    def call(self, inputs):
        return self.infer_text(inputs[0], inputs[1])

    def cartesian(self, inputs):
        n = tf.shape(inputs[0])[0]
        hypothesis = tf.tile(inputs[0], tf.shape(inputs[1]))
        premise = tf.repeat(inputs[1], n)
        probs = self.infer_text(hypothesis, premise)
        return tf.reshape(probs, [-1, n])


# Build servable model.
servable = Servable()
signatures = dict()


# Define the serving_default signature.
@tf.function(
    input_signature=[
        [
            tf.TensorSpec([None], tf.string, "hypothesis"),
            tf.TensorSpec([None], tf.string, "premise"),
        ]
    ]
)
def serving_default(inputs):
    outputs = dict()
    outputs["entailment"] = servable(inputs)
    return outputs


signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = serving_default


# Define the cartesian signature.
@tf.function(
    input_signature=[
        [
            tf.TensorSpec([None], tf.string, "hypothesis"),
            tf.TensorSpec([None], tf.string, "premise"),
        ]
    ]
)
def cartesian(inputs):
    outputs = dict()
    outputs["entailment"] = servable.cartesian(inputs)
    return outputs


signatures["cartesian"] = cartesian

# Save the servable model and create the assets.extra directory.
save_dir = "servables/barba/1"
tf.saved_model.save(servable, save_dir, signatures=signatures)
extra_dir = os.path.join(save_dir, "assets.extra")
os.makedirs(extra_dir, exist_ok=True)

# Prepare warmup requests.
hypotheses = tf.constant(
    [
        "A robot should protect human",
        "机器人应当保护人类",
        "ロボットは人間を守らなければならない",
        "로봇은 인간을 보호해야 한다",
    ],
    dtype=tf.string,
)
premises = tf.constant(
    [
        "A robot may not injure a human being or, through inaction, "
        "allow a human being to come to harm",
        "机器人不得伤害人类，或坐视人类受到伤害",
        "ロボットは人を傷つけたり、人に危害を加えたりしてはならない",
        "로봇은 인간을 다치게 하거나 인간이 해를 입도록 허용해서는 안 됩니다",
    ],
    dtype=tf.string,
)

# Write warmup requests to file.
warmup_record_path = os.path.join(extra_dir, "tf_serving_warmup_requests")
with tf.io.TFRecordWriter(warmup_record_path) as writer:
    request = predict_pb2.PredictRequest()
    signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.model_spec.name = tf.saved_model.SERVING
    request.model_spec.signature_name = signature_name
    request.inputs["hypothesis"].CopyFrom(tf.make_tensor_proto(hypotheses))
    request.inputs["premise"].CopyFrom(tf.make_tensor_proto(premises))
    predict_log = prediction_log_pb2.PredictLog(request=request)
    log = prediction_log_pb2.PredictionLog(predict_log=predict_log)
    writer.write(log.SerializeToString())

# Embed batching parameters.
batching_parameters_path = os.path.join(extra_dir, "batching_parameters.pbtxt")
with open(batching_parameters_path, "w") as file:
    file.write("max_batch_size { value: 32 }\n")
    file.write("batch_timeout_micros { value: 2000 }\n")
    file.write("num_batch_threads { value: 8 }\n")
    file.write("max_enqueued_batches { value: 100 }\n")
