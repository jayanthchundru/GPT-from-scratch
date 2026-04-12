# GPT

### Building Transformer based language model from scratch.

##### 1. This repo contains from the scratch implementation of GPT - Architecture 

##### 2. For the tokenizer - We have used the GPT pre-trained tokenizer (Tiktoken) which is built based on the Byte Pair Encoding (BPE)

##### 3. attention.py file contains different variants of attentions for practice like SelfAttention, CausalSelfAttention,  MultiHeadAttention ( which is used in our Transformers.py for the mini-GPT.py)

##### 4. Transformers.py contains each building block of the transformer block.

##### 5. mini_GPT.py uses the transformer block and add the MLP layer and function to generate text  ( Next token Prediction)

##### 6. Soon I will add the post training code for the mini_GPT.py

##### Resources: 
###### 1. Viszura's Building LLMs from Scratch : 
https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&si=8EH0EpTQ1L_ft1W8
###### 2. Andrew Karpathy Nano GPT : 
https://youtu.be/kCc8FmEb1nY?si=pLnpKxcVWVqTq2Di