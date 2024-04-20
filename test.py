import os
import torch
from retrieval.retrieval import Memory
from sentence_transformers import SentenceTransformer
import openai

API_KEY = "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"
MODEL = "gpt-3.5-turbo"
openai.api_key = API_KEY

def is_api_key_valid():
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role":"user", "content": "Hello"}],
        temperature=0,
        max_tokens=50
    )
    print(response)

    
if __name__=='__main__':

    print(is_api_key_valid())