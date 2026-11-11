import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
import random
import tiktoken
import time
import math
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

@dataclass
class GPTConfig:
    n_embd: int = 768
    n_head: int = 12
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    dropout: float = 0.0
    bias: bool = True

class MLP(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.embd = config.n_embd
        self.head = config.n_head
        self.ln_1 = nn.LayerNorm(self.embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(self.embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd,self.n_embd)
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.shape
        assert C % self.n_head == 0
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head,).transpose(1,2)
        k = k.view(B, T, self.n_head, C // self.n_head,).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head,).transpose(1,2)

        y = F.scaled_dot_product_attention(q,k,v)
        # wei = q @ k.transpose(-2,-1) * (C // self.n_head) ** -0.5
        # wei = wei.masked_fill(self.bias[:,:,:T,:T] == 0,float("-inf"))
        # wei = F.softmax(wei,dim=-1)
        # y = wei @ v

        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__() 
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.init_weight)
    def init_weight(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T,)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B,T,C)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.shape[-1]), targets.view(-1))
        return logits, loss

    def generate(self, idx):
        pass

class DataLoader:
    """从数据集中加载数据，用分词器encode后，可供batch获取"""
    def __init__(self, B, T,):
        self.B = B
        self.T = T
        self.enc = tiktoken.get_encoding("gpt2")
        self.data = ''
        self.token_len = 0

    def load_data(self,path):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = self.enc.encode(text)
        self.token_len = len(tokens)
        print(f'the length of tokens is {len(tokens)}')
        print(f'finish a loop needs {len(tokens) // (B *T) + 1} batches')
        self.data = torch.tensor(tokens, dtype=torch.long)

    def get_batch(self):
        start_idx = random.randint(0, self.token_len - B * T -1)
        batch = self.data[start_idx: start_idx + B * T + 1]
        x = batch[:-1].view(B, T)
        y = batch[1:].view(B, T)
        return x,y

num_return_sequences = 5
max_length = 50
""" model = GPT.from_pretrained('gpt2') """
model = GPT(GPTConfig(vocab_size=50304))
model.to('cuda')



config = GPTConfig()
enc = tiktoken.get_encoding("gpt2")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_en = os.path.join(BASE_DIR,'..','data','input.txt')
""" with open(path_en,'r',encoding='utf-8') as f:
    text = f.read() """
#tokens = enc.encode(text)


B, T = 8, 512
data_loader = DataLoader(B, T)
data_loader.load_data(path_en)
optimizer = torch.optim.AdamW(model.parameters(),lr=6e-4,betas=(0.9, 0.95),eps=1e-8)

max_lr = 6e-4
min_lr = 6e-5
max_steps = 5000
warm_up_steps = 100
def get_lr(step):
    if step < warm_up_steps:
        return max_lr * step / warm_up_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warm_up_steps) / (max_steps - warm_up_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


model.train()
torch.set_float32_matmul_precision('high') 
#model = torch.compile(model)

for step in range(max_steps):
    t0 = time.time()
    x, y = data_loader.get_batch()
    x, y = x.to('cuda'), y.to('cuda')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step + 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    if (step+1) % 20 == 0:
        print(f"step{step+1}: loss = {loss.item():.6f} | lr = {lr:.4f} | norm = {norm:.4f} | time = {(t1-t0) * 1000:6f}ms")

tokens_gen = enc.encode("CORIOLANUS:")
tokens_gen = torch.tensor(tokens_gen, dtype=torch.long, device='cuda').unsqueeze(0)
tokens_gen = tokens_gen.repeat(num_return_sequences, 1)
x_gen = tokens_gen.to('cuda')

model.eval()
while x_gen.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x_gen)
        # print(logits.shape)
        last_logits = logits[:, -1 , :]
        probs = F.softmax(last_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs,k=50,dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(topk_probs, num_samples=1)
        next_token = torch.gather(topk_indices, -1, next_token)
        x_gen = torch.cat((x_gen, next_token), dim=1)

for i in range(num_return_sequences):
    tokens = x_gen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)