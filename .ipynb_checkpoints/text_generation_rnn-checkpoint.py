import os
import warnings
import time
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

## Download Data
path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)

## Read the data
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
print(f"Length of text: {len(text)} characters")

print(text[:250])
vocab = sorted(set(text))
print(f"{len(vocab)} unique characters")

## Process the data. Note below text are only for examples and testing ids_from_chars and chars_from_ids etc.
example_texts = ["abcdefg", "xyz"]

# TODO 1
chars = tf.strings.unicode_split(example_texts, input_encoding="UTF-8")
print(chars)

#Layer to retrieve ids from chars
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None
)
ids = ids_from_chars(chars)
print(ids)

#Layer to retrieve chars from ids
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)

#chars = chars_from_ids(ids)
#print(chars)
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

#print(text_from_ids(ids))

## Predict the data. Note here we are passing original text
all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode("utf-8"))

