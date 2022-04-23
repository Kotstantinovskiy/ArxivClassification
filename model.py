import streamlit as st
from torch import nn
import numpy as np
from transformers import DistilBertForSequenceClassification


class ArxivModel:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.model.to('cpu')

    def get_logits(self, tweet_text):
        text_tokens = self.tokenizer(tweet_text, return_tensors="pt").to('cpu')
        softmax = nn.Softmax(dim=1)

        return softmax(self.model(**text_tokens).logits.detach()).numpy()[0]

    def get_idx_class(self, tweet_text, thr=-1.0):
        logits = self.get_logits(tweet_text)

        if thr == -1.0:
            return [(np.argmax(logits), np.max(logits))]
        else:
            sum_probs = 0.0
            idxs = []
            for p in np.argsort(logits)[::-1]:
                sum_probs += logits[p]
                idxs.append((p, logits[p]))
                if sum_probs > thr:
                    return idxs


@st.cache
def load_model(path="./checkpoint-15500", num_labels=153):
    return DistilBertForSequenceClassification.from_pretrained(path, num_labels=num_labels)
