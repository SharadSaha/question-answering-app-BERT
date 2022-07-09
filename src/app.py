from distutils.command.config import config
import os
import yaml
import argparse
import numpy as np
import tensorflow as tf
from model import read_params,get_FineTunedBERT
from get_model import load_BERT_model

max_seq_length = 399

from tokenizers import BertWordPieceTokenizer

#############################################
args = argparse.ArgumentParser()
args.add_argument("--config",default="../params.yaml")
parsed_args = args.parse_args()
config_path = parsed_args.config
model,vocab_file = load_BERT_model(config_path)
#############################################


def load_tokenizer(vocab_file):
  tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=True)
  return tokenizer

class Sample:
  def __init__(self, question, context, start_char_idx=None, answer_text=None, all_answers=None):
    self.question = question
    self.context = context
    self.start_char_idx = start_char_idx
    self.end_char_idx = -1
    self.answer_text = answer_text
    self.skip = False
    self.start_token_idx = -1
    self.end_token_idx = -1
    self.max_seq_length = max_seq_length
    self.padding_length = 10
    self.all_answers = all_answers

  def get_tokens(self):
    context = " ".join(str(self.context).split())
    question = " ".join(str(self.question).split())
    tokenizer = load_tokenizer(vocab_file)
    tokenized_context = tokenizer.encode(context)
    tokenized_question = tokenizer.encode(question)

    return (context,question),(tokenized_context,tokenized_question)

  def get_ids(self,tokenized_context,tokenized_question):
    input_ids = tokenized_context.ids + tokenized_question.ids[1:]
    seg_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
    mask = [1] * len(input_ids)
    self.padding_length = self.max_seq_length - len(input_ids)
    return (input_ids,seg_ids,mask)

  def preprocess(self):

    # getting the tokenized text
    (context,question),(tokenized_context,tokenized_question) = self.get_tokens()

    if self.answer_text is not None:
      answer = " ".join(str(self.answer_text).split())

      # calculating end character index
      self.end_char_idx = self.start_char_idx + len(answer)
      if self.end_char_idx >= len(context):
          self.skip = True
          return
    
      is_char_in_ans = [0] * len(context)
      for idx in range(self.start_char_idx, self.end_char_idx):
          is_char_in_ans[idx] = 1
      ans_token_idx = []

      # finding the relevant tokens present in the answer
      for idx, (start, end) in enumerate(tokenized_context.offsets):
          if sum(is_char_in_ans[start:end]) > 0:
              ans_token_idx.append(idx)
      if len(ans_token_idx) == 0:
          self.skip = True
          return

      self.start_token_idx = ans_token_idx[0]
      self.end_token_idx = ans_token_idx[-1]

    # getting the ids necessary for BERT input
    (input_ids,seg_ids,mask) = self.get_ids(tokenized_context,tokenized_question)

    # adding necessary padding 
    if self.padding_length > 0:
        input_ids = input_ids + ([0] * self.padding_length)
        mask = mask + ([0] * self.padding_length)
        seg_ids = seg_ids + ([0] * self.padding_length)
    elif self.padding_length < 0:
        self.skip = True
        return

    self.input_word_ids = input_ids
    self.segment_ids = seg_ids
    self.input_mask = mask
    self.context_token_to_char = tokenized_context.offsets



def create_examples(data):
    examples = []
    for item in data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qas in para["qas"]:
                question = qas["question"]
                if "answers" in qas:
                    answer_text = qas["answers"][0]["text"]
                    start_char_idx = qas["answers"][0]["answer_start"]
                    all_answers = [_["text"] for _ in qas["answers"]]
                    sample = Sample(question, context, start_char_idx, answer_text,all_answers)
                else:
                    sample = Sample(question, context)

                # preprocess each sample
                sample.preprocess()
                examples.append(sample)
    return examples

def create_data_target_pairs(examples):
    dataset_dict = {
        "input_word_ids": [],
        "segment_ids": [],
        "input_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
    x = [dataset_dict["input_word_ids"],
         dataset_dict["input_mask"],
         dataset_dict["segment_ids"]]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


def get_predicted_answers(data):
    test_samples = create_examples(data)
    x_test, _ = create_data_target_pairs(test_samples)
    pred_start, pred_end = model.predict(x_test)
    answers = []
    questions = []
    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        test_sample = test_samples[idx]
        offsets = test_sample.context_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
        pred_ans = None
        if start >= len(offsets):
            continue
        pred_char_start = offsets[start][0]
        if end < len(offsets):
            pred_ans = test_sample.context[pred_char_start:offsets[end][1]]
        else:
            pred_ans = test_sample.context[pred_char_start:]
        questions.append(test_sample.question)
        answers.append(pred_ans)
    return questions,answers


def display_question_answers(questions,answers):
  for question,answer in zip(questions,answers):
    print("Q: " + question)
    print("A: " + answer)



c1 = '''Shah Rukh Khan (pronounced [ˈʃɑːɦɾʊx xɑːn]; born 2 November 1965), also known by the initialism SRK, is an Indian actor, film producer, and television personality who works in Hindi films. Referred to in the media as the "Baadshah of Bollywood" (in reference to his 1999 film Baadshah), "King of Bollywood" and "King Khan", he has appeared in more than 80 films, and earned numerous accolades, including 14 Filmfare Awards. The Government of India has awarded him the Padma Shri, and the Government of France has awarded him the Ordre des Arts et des Lettres and the Legion of Honour. Khan has a significant following in Asia and the Indian diaspora worldwide. In terms of audience size and income, he has been described as one of the most successful film stars in the world.'''
data1 = {"data":
    [
        {"title": "Shah Rukh Khan",
         "paragraphs": [
             {
                 "context": c1,
                 "qas": [
                     {"question": "What name is Shah Rukh Khan refered to the media as?",
                      "id": "Q1"
                      },
                     {"question": "How many films did he appear in?",
                      "id": "Q2"
                      }
                 ]}]}]}

questions,answers = get_predicted_answers(data1)
display_question_answers(questions,answers)