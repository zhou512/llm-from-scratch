# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


batch_size = 64
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
max_iters = 60000

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
def estimate_loss():
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


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets = None):
        logits = self.token_embedding_table(idx) #(4,8,65)
        B,T,C = logits.shape
        logits_flat = logits.view(B*T,C)
        if targets is not None:
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits_flat,targets)
        else:
            loss = None
        return logits,loss

    def generate(self,idx,max_tokens):
        for _ in range(max_tokens):
            logits,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=1)
            idx_next = torch.multinomial(probs,num_samples=1) #4*1
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
  


model = BigramLanguageModel(vocab_en_size)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(),lr=3e-4)

# %%
for i in range(max_iters):
    if(i % eval_iters == 0):
        out = estimate_loss()
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




