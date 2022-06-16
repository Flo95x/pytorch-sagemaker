import os
import json
import torch
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Union, Tuple
from transformers import AutoTokenizer, AutoModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast

device = "cpu"

# Helper: Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def model_fn(model_dir: str):
    # init model and tokenizer with files of model.tar.gz
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)

    # make sure model is on correct device, i.e. cpu
    model.to(device).eval()
    return model, tokenizer

def input_fn(request_body: Union[str, bytes], request_content_type: str):
    assert request_content_type=='application/json'
    text_list = json.loads(request_body)['inputs']
    return text_list

def predict_fn(sentences: List[str], model_and_tokenizer: Tuple[XLMRobertaModel, XLMRobertaTokenizerFast]):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings

def output_fn(sentence_embeddings: torch.Tensor, content_type: str):
    assert content_type=='application/json'
    res = sentence_embeddings.tolist()
    # return a json object
    return json.dumps(res)
