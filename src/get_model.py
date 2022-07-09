from distutils.command.config import config
import os
import yaml
import argparse
import numpy as np
import tensorflow as tf
from model import read_params,get_FineTunedBERT

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def load_BERT_model(config_path):
    config = read_params(config_path)
    weights_path = "../" + os.path.join(config['model_dir'],'weights.h5')
    print("Model path : ",weights_path)
    model,vocab_file = get_FineTunedBERT()
    model.load_weights(weights_path)
    return model,vocab_file



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="../params.yaml")
    parsed_args = args.parse_args()
    model,vocab_file = load_BERT_model(config_path=parsed_args.config)