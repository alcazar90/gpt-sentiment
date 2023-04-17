"""
This script is used to prepare the data for the extended_by_char dataset.

The extended_by_char dataset is the same as the by_char dataset, except that it
containts an additional text from 1000 post from reddit-chile
"""

import os
import pickle
import pandas as pd
import numpy as np

# open datasets
train_df = pd.read_csv('./data/train.tsv', sep='\t')
reddit = pd.read_csv("./data/20230415_api_reddit.csv")

# concatenate title and text from reddit observations without NA in text 
full_text = reddit[~reddit.text.isna()]
full_text['concat_text'] = full_text['title'] + ' ' + full_text['text']

# create a single string with all the text
full_data = ''.join([t for t in full_text.concat_text])

# create a single string with all the titles
partial_text = reddit[reddit.text.isna()]['title']
partial_data = '\n'.join([t for t in partial_text])

# concatenate all text from reddit into a single big document
reddit_data = full_data + ' ' + partial_data

# create a single string with all the text from train (tweets)
data = ''.join([t for t in train_df.texto])

# concatenate all text from tweets and reddit into a single big document
augmented_data = data + ' ' + reddit_data

# get all the unique characters that occur in this text
chars = sorted(list(set(augmented_data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from character to integers
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

def encode(s):
    """encoder: take a string, return a list of integers"""
    return [stoi[c] for c in s]

def decode(l):
    """decoder: take a list of integers, return a string"""
    return ''.join([itos[i] for i in l])

# create the train and val splits
n = len(augmented_data)
train_data = augmented_data[:int(n*0.9)]
val_data = augmented_data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile("./data/extended_by_char-vs2/train.bin")
val_ids.tofile("./data/extended_by_char-vs2/val.bin")

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open("./data/extended_by_char-vs2/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

