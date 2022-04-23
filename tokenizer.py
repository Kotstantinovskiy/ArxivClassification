import streamlit as st
from transformers import DistilBertTokenizerFast


@st.cache(allow_output_mutation=True)
def get_tokenizer(num_labels=153):
    return DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
