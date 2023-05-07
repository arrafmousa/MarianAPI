import os
from os import listdir
from os.path import isfile, join
from transformers import AutoTokenizer


class MarianAPITokenizer:
    def __init__(self, base_model="AhmedSSoliman/MarianCG-CoNaLa-Large"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model,
                                                       use_fast=True,
                                                       padding='max_length',
                                                       model_max_length=512)

    # def train(self, corpus_dir: str):
    #     def clean(example):
    #         return example
    #
    #     files = [f for f in listdir(corpus_dir) if isfile(join(corpus_dir, f))]
    #     texts = []
    #     for file in files:
    #         with open(os.path.join(corpus_dir, file), encoding='utf8') as f:
    #             texts += [clean(example) for example in f.readlines()]
    #     self.tokenizer = self.tokenizer.train_new_from_iterator(texts,5000)
    #     self.tokenizer.save()

    def tokenize(self,sentence):
        return self.tokenizer.tokenize(sentence)


a = MarianAPITokenizer("gpt2")
print(a.tokenize("hello world"))
