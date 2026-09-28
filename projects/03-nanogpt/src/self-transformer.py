# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


batch_size = 64
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500
max_iters = 5000
n_embed = 384
num_heads = 6
n_layer = 6
dropout = 0.2
lr = 3e-4

torch.manual_seed(1337)
print(f"device : {device}")

def decode(data:list):
    res = ""
    for num in data:
        res += itos[num]
    return res

def encode(data:str):
    return [stoi[c] for c in data]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_en =  os.path.join(BASE_DIR,'..','data','input.txt')

with open(path_en,"r",encoding='utf-8') as f:
    text_en = f.read()
print(f"text_en:{len(text_en)}")


""" with open("../data/input_zh.txt","r",encoding='utf-8') as f:
    text_zh = f.read()
print(f"text_zh:{len(text_zh)}")
chars_zh = sorted(list(set(text_zh)))
vocab_zh_size = len(chars_zh) """

chars_en = sorted(list(set(text_en)))
vocab_en_size = len(chars_en)

""" zhs = ''.join(chars_zh) """

stoi = {s:i for i,s in enumerate(chars_en)}
itos = {i:s for s,i in stoi.items()}
data = torch.tensor([stoi[c] for c in text_en])


def build_three_dataset(data):
    n1 = int(len(data) * 0.8)
    n2 = int(len(data) * 0.9)
    return data[:n1],data[n1:n2],data[n2:]

train_data,dev_data,test_data = build_three_dataset(data)
set_dict = {"train":train_data,"dev":dev_data,"test":test_data}



def get_batch(split:str):
    data = set_dict[split]
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y
xb,yb = get_batch('train')

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for split in ["train","dev"]:
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * (n_embed // num_heads) ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        wei = F.softmax(wei,dim=2)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiAttentionHeads(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed,n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Block(nn.Module):
    def __init__(self, num_heads, n_embed):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiAttentionHeads(num_heads,head_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table = nn.Embedding(block_size,n_embed)
        #self.sa_heads = MultiAttentionHeads(4,n_embed // 4)
        #self.ffwd = FeedForward(n_embed)
        self.blcs = nn.Sequential(
           *[Block(num_heads,n_embed) for _ in range(n_layer)],
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed,vocab_size)

    def forward(self,idx,targets = None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)         #B,T,n_embed
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb       #B,T,n_embed
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blcs(x)
        logits = self.lm_head(x)    #B,T,vocab_size
        if targets is not None:
            B,T,C = logits.shape
            logits_flat = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits_flat,targets)
        else:
            loss = None
        return logits,loss

    def generate(self,idx,max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:,-block_size:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=1)
            idx_next = torch.multinomial(probs,num_samples=1) #4*1
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
  


model = TransformerLanguageModel(vocab_en_size)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(),lr=lr)

for i in range(max_iters):
    if(i % eval_iters == 0):
        out = estimate_loss(m)
        print(f"step {i} training loss = {out['train']:.4f}, dev loss = {out['dev']:.4f}")
    xb,yb = get_batch('train')
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss)

# %%
context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(idx=context,max_tokens=100)[0].tolist()))




