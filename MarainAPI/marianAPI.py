import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class MarianAPI:
    """
    1. Use the MarianCG model to create the skeleton code to be run, but with the arguments *masked*.
    2. Train another model to detect the arguments, this model can be another transformer - or even a gpt-2

    Example :

    [context] :
        water evaporates at the rate of 1 gram per second, how much time do we need to create 10 grams of vapor

    [masked code] :
        vapor = 0 [grams]
        time = 0 [seconds]
        while vapor < >>token1<< :
            time += >>token2<<
        return time

    [target tokens]:
        >>token1<< : 10 [grams]
        >>token2<< : 1 [second]

    ** For now I have an end-to-end model that generates the code and uses APIs with its perspective arguments
    TODO : reformat the APIs detected in the pulled code from 'github'
    TODO : mask the API arguments with tokens and insert them into a dictionary to be passed to the other model to query them
    TODO : train another model to detect the arguments ** from our dataset ** because we need the context


    -------------------------- Order Of Things ---------------------
    1. MarianCG : (or equal programmer)
        1.1. Instantiate the model                                                                | 1.5 V
        1.2. Mask the APIs' arguments, in pubchempy corpus and in SimQA dataset                   | 7.5 V
        1.3. Finetune the model on our masked data (1.2) - familiarizing the model with the APIs  | 7.5
        1.4. test, and save !                                                                     | 7.5
    2. Argument manager :                                                                         | 21.5
        2.1. create another model capable of taking the following arguments:                      | 21.5
                * context & question
                * generated masked code
            and return the following:
                * dictionary mapping the masks to the arguments
        2.2. train the model on unmasked SimQA                                                    | 21.5
        2.3. test, and save !                                                                     | 21.5
    3. MarianAPI :                                                                                | 28.5
        3.1. load (1.4) and (2.3)                                                                 | 28.5
        3.2. wrap them in a single model.                                                         | 28.5
        3.3. train them  in parallel. (???)                                                       |
    """

    class MarianAPIDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

    def __init__(self, base_model="AhmedSSoliman/MarianCG-CoNaLa-Large"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model, max_length=512)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model,
                                                       use_fast=True,
                                                       padding='max_length',
                                                       model_max_length=512)

    def train(self, corpus_path: str):
        with open(corpus_path) as corpus:
            texts = corpus.readlines()

        # train tokenizer
