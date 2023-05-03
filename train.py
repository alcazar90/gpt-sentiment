"""
Training script to run the model.

Include logging with weights and biases.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig

# ------------------------------------------------------------------------------
# I/O
out_dir = 'out/extended_by_char_out'
eval_interval = 250   # eval every eval_interval iterations (micro steps or iters when gradient_accumulation_steps > 1)
log_interval = 30
eval_iters = 200
eval_only = False   # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = 'scratch'   # 'scratch' or 'resume' or 'gpt2'

# logging
wandb_log = True
wandb_project = 'sentiment_gpt_esp'
wandb_run_name = 'gpt2-sentiment-' + time.strftime("%Y-%m-%d-%H:%M:%S")
num_samples = 3
max_new_tokens = 140
temperature = 0.9

# data
dataset = 'by_char'     # extended_by_char (data augmentation with reddit posts)
gradient_accumulation_steps = 4 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256  # what is the maximum context length for predictions?

# model
n_layer = 6 # how many attentions blocks to stack
n_head = 6  # number of attentions heads per layer
n_embd = 384   # embedding size, should be divisible by n_head (e.g. 384/6 = 64)
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 3e-4    # max learning rate
max_iters = 3000         # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True    # whether to decay the learning rate
warmup_iters = 250 # how many steps to warm up forl
lr_decay_iters = 2700 # should be ~= max_iters per Chinchilla
min_lr = 1e-5   # minimmum learning rate, should be ~= learning_rate / 10 per Chinchilla

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False   # use PyTorch 2.0 to compile the model to be faster
# ------------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
# ------------------------------------------------------------------------------

# we are running on a single gpu, and one process
master_process = True
seed_offset = 0
gradient_accumulation_steps *= 4  # simulate 8 gpus

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(33313988 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load dataset and define data loader
train_data = np.memmap(f"./data/{dataset}/train.bin", dtype=np.uint16, mode='r')
val_data = np.memmap(f"./data/{dataset}/val.bin", dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # TODO: check pin_memory()
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


iter_num = 0
best_val_loss = 1e9
vocab_size = None

# attempt to derive coab_size from the dataset
meta_path = os.path.join(f"./data/{dataset}/meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):

    with open(f"./data/{dataset}/meta.pkl", 'rb') as f:
        meta = pickle.load(f)

    meta_vocab_size = meta['vocab_size']
    vocab_size = meta['vocab_size']

    itos = meta['itos']
    stoi = meta['stoi']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    def encode(s):
        """encoder: take a string, output a list of integers"""
        return [stoi[c] for c in s]
    
    def decode(l):
        """decoder: take a list of integers, output a string"""
        return ''.join([itos[i] for i in l])

else:
    import tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')

    def encode(s):
        """encoder: take a string, output a list of integers"""
        return tokenizer.encode_ordinary(s)

    def decode(l):
        """decoder: take a list of integers, output a string"""
        return tokenizer.decode(l)

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                  vocab_size=vocab_size, block_size=block_size, dropout=dropout,
                  bias=bias,)   # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume
    # training the rest of the attributes (e.g. dropout) can stay as desired from
    # command line 
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# TODO: add others initialization methods ('resume')
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2),
                                       device_type)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# function to get samples from the model, for logging in WandB
@torch.no_grad()
def get_samples():
    model.eval()
    out = []
    for k in range(num_samples):
            y = model.generate(idx=torch.zeros((1, block_size), dtype=torch.long, device=device), max_new_tokens=max_new_tokens, temperature=temperature)
            out.append(f"({k+1}) {decode(y[0][block_size:].tolist())}")
    model.train()
    return '\n'.join(out)

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    table = wandb.Table(columns=["iter_num", "samples", "train_loss", "val_loss"])

# training loop
X, Y = get_batch('train')   # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:

    # determine and set learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        samples_to_save = get_samples()
        print('samples:\n', samples_to_save)
        print('----------------------------------------')
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                })

            table.add_data(iter_num, samples_to_save, losses['train'], losses['val'])

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate
    # larget batch size and using GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
        # inmediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f" -> iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        wandb.log({"sample_table": table}, commit=False)
        break
