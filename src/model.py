from distutils.command.config import config
import yaml
import os
import argparse
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from keras import layers
import keras

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def load_BERT(url,max_seq_length,trainable=True):
  input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='segment_ids')

  bert_layer = hub.KerasLayer(url, trainable=True)

  # pooled output has shape (batch size, embedding dim) which is an embedding of the [CLS] token and represents entire sequence
  # sequence output has shape (batch size, max sequence length, embedding dim) which has representation for each token 

  pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
  vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
  to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

  return vocab_file,bert_layer,pooled_output,sequence_output,input_word_ids, input_mask, segment_ids


class ModelParams:
  def __init__(self,learning_rate=1e-5, beta_1=0.9, beta_2=0.98, epsilon=1e-9):
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    self.learning_rate = learning_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon

def get_FineTunedBERT():

    url = "https://hub.tensorflow.google.cn/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
    max_seq_length = 399
    vocab_file,bert_layer,pooled_output,sequence_output,input_word_ids, input_mask, segment_ids = load_BERT(url,max_seq_length,True)
    params = ModelParams()
    # start and end logits
    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(sequence_output)
    start_logits = layers.Flatten()(start_logits)
    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(sequence_output)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)
    model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[start_probs, end_probs])
    loss = params.loss
    optimizer = params.optimizer
    model.compile(optimizer=optimizer, loss=[loss, loss])

    return model,vocab_file
    


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    model = get_FineTunedBERT()
    print(model.summary())

