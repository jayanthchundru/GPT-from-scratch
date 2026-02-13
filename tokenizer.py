import importlib.metadata
import re
import requests
import os
import matplotlib.pyplot as plt 
import tiktoken 
import importlib

if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    )
    file_path = "the-verdict.txt"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)


with open("the-verdict.txt", "r") as f:
    raw_text = f.read()

# print("Total number of characters : ", len(raw_text))
# print(raw_text[:99])



preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(f"Length of the preprocessed text : {len(preprocessed)}")
# print(preprocessed[:40])

unique_words = sorted(set(preprocessed))
vocab_size = len(unique_words)
unique_words.extend(["<|endoftext|>", "<|unk|>"])

# print(f"The vocabulary size is : {vocab_size}")

vocab = {token : integer for integer, token in enumerate(unique_words)}

# print(len(vocab.items()))

# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)
#     if i > 50:
#         break


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i : s for s , i in self.str_to_int.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids 
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([.,?!"()\'])', r"\1", text)
        return text


tokenizer = SimpleTokenizerV1(vocab)

text1 = 'Hello! do you like to have tea ?'
text2 = 'In the sunlit terraces of the place'

text = " <|endoftext|> ".join((text1, text2))
ids = tokenizer.encode(text)
decoded_text = tokenizer.decode(ids)

# print(ids)
# print(decoded_text)

''' Types of special tokens used are BOS - Begin of Sentence , EOS - End of Sentence, 
PAD  - Padding in LLMs <|endoftext|> is the ending of text '''


print("Version of tiktoken :", importlib.metadata.version("tiktoken"))

bpe_tokenizer = tiktoken.get_encoding('gpt2')


text = ( " Hello , do you like to have tea ? <|endoftext|> In the sunlit places of someunknownplace." )
ids = bpe_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(ids)

strings = bpe_tokenizer.decode(ids)
# print(strings)


# -------------------------- Creating Input-Output Pairs ------------------------------- 

enc_text = bpe_tokenizer.encode(raw_text)

enc_sample = enc_text[50:]

context_size = 4 

x = enc_sample[:context_size]
y = enc_sample[1: context_size+1]

# print(x)
# print(y)


# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(f"context :{bpe_tokenizer.decode(context)} ----> desired output : {bpe_tokenizer.decode([desired])}")



import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        for i in range(1, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i+1 : i+ max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )
    
    return dataloader



# data_loader = create_dataloader_v1(
#     raw_text, batch_size=4, max_length=4, stride=1, shuffle=False
# )

# data_iter = iter(data_loader)

# first_batch = next(data_iter)
# print(first_batch)



#### --------------------- Embeddings Layer --------------------------------------
import torch
from torch import nn


vocab_size = 50257
output_dim = 256

embedding_layer = nn.Embedding(vocab_size, output_dim)

print("Embedding layer shape : ",embedding_layer.weight.shape)


# ------ single token embeddings --------
# print(embedding_layer(torch.tensor([5])))

# ------ Multiple token embeddings --------
# print(embedding_layer(torch.tensor([5, 6, 3, 3])))



### ------------------ Positional Encoding --------------------------------

''' 
    There are two types of Positional embeddings 
    1. Absolute Embeddings -- This encodes the exact position of the token in the sentence  -- order of sequence generation is important !! 
    Initially OpenAi and the Attention is all you need paper was used absolute embeddings 
        
    2. Relative Embeddings - this measures position relative to the tokens in the sentence -- 
    this is much feasible / suitable for the longer sequence lengths ( like Language modelling tasks)
    
'''


max_length = 4

data_loader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

data_iter = iter(data_loader)

inputs, outputs =  next(data_iter)

token_embeddings = embedding_layer(inputs)
print("Token Embeddings Shape : ", token_embeddings.shape)


context_length = max_length

pos_embedding_layer = nn.Embedding(context_length, output_dim)

print("POS Embedding Layer : ", pos_embedding_layer.weight.shape)

