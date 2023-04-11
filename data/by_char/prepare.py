"""
Prepare the dataset for character-level language modeling.
So instead of encoding with a fancy tokenizer (GPT-2 BPE tokens), we 
just map characters to ints. Will save train.bin, val.bin containing
the ids, and meta.pkl containing the encoder and decoder and some other
related info.
"""
import os
import pickle
import numpy as np


# open train.tsv file and only read the second column of each line
with open("./data/train.tsv", "r") as f:
    data = f.read()
    data = data.split('\n')
    data = [row.split('\t') for row in data][:-1]
    data = [row[1] for row in data[1:] if len(row) == 3]
    data = ''.join(data)


print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from character to integers
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

def encode(s):
    """encoder: take a string, output a list of integers"""
    return [stoi[c] for c in s]

def decode(l):
    """decoder: take a list of integers, output a string"""
    return ''.join([itos[i] for i in l])


# create the train and val splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile("./data/by_char/train.bin")
val_ids.tofile("./data/by_char/val.bin")

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open('./data/by_char/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
