import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn import functional as F
import random
CHAR_SIZE = 27
CHAR_VECTOR_SIZE = 10
BLOCK_SIZE = 3
W1_SIZE = 200
BATCH_SIZE = 128


def build_dataset(data:list):
    X = []
    Y = []
    blocksize = 3
    for name in data:
        context = blocksize * [0]
        for c in name:
            ix = ctoi[c]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

names = open("names.txt","r").read().splitlines()
chars = sorted(set("".join(names)))
ctoi = {c:i+1 for i,c in enumerate(chars)}
ctoi["."] = 0
itoc = {i:c for c,i in ctoi.items()}
for i in range(len(names)):
    names[i] = '.' + names[i] + '.'



random.seed(42)
random.shuffle(names)
n1 = int(len(names) * 0.8)
n2 = int(len(names) * 0.9)

Xtr,Ytr = build_dataset(names[:n1])
Xdv,Ydv = build_dataset(names[n1:n2])
Xts,Yts = build_dataset(names[n2:])

C = torch.randn((CHAR_SIZE,CHAR_VECTOR_SIZE))   #将每个字符的索引对应到成一个向量
W1 = torch.randn((CHAR_VECTOR_SIZE * BLOCK_SIZE,W1_SIZE))
B1 = torch.randn(W1_SIZE)
W2 = torch.randn((W1_SIZE,CHAR_SIZE))
B2 = torch.randn(CHAR_SIZE)
parameters = [C,W1,W2,B1,B2]

for p in parameters:
    p.requires_grad = True

losses = []
lrses = []
# training set
for i in range(100000):
    lr = 0.85 if i < 50000 else 0.085
    for p in parameters:
        p.grad = None
    ix = torch.randint(0,Xtr.shape[0],(64,))
    emb = C[Xtr[ix]]       
    act = torch.tanh(emb.view(-1,CHAR_VECTOR_SIZE * BLOCK_SIZE) @ W1 + B1)
    logits = act @ W2 + B2
    loss = F.cross_entropy(logits,Ytr[ix])
    loss.backward()
    losses.append(loss.item())
    lrses.append(lr)
    for p in parameters:
        p.data -= lr * p.grad

# plt.plot(lrses,losses)
# plt.show()
# print(f"loss = {loss}")

emb = C[Xtr]
act = torch.tanh(emb.view(-1,CHAR_VECTOR_SIZE * BLOCK_SIZE) @ W1 + B1)
logits = act @ W2 + B2
loss = F.cross_entropy(logits,Ytr)
print(f"training loss = {loss}")

emb = C[Xdv]
act = torch.tanh(emb.view(-1,CHAR_VECTOR_SIZE * BLOCK_SIZE) @ W1 + B1)
logits = act @ W2 + B2
loss = F.cross_entropy(logits,Ydv)
print(f"dev loss = {loss}")

emb = C[Xts]
act = torch.tanh(emb.view(-1,CHAR_VECTOR_SIZE * BLOCK_SIZE) @ W1 + B1)
logits = act @ W2 + B2
loss = F.cross_entropy(logits,Yts)
print(f"test loss = {loss}")

g = torch.Generator().manual_seed(11468)
for _ in range(20):
    out = []
    context = [0] * BLOCK_SIZE
    while True:
        emb = C[torch.tensor(context)] #(1,3,10)
        act = torch.tanh(emb.view(1,-1) @ W1 + B1)
        logits = act @ W2 + B2
        probs = F.softmax(logits,dim=1)
        ix = torch.multinomial(probs,num_samples=1,replacement=True,generator=g).item()
        context = context[1:] + [ix]
        out.append(itoc[ix])
        if ix == 0:
            if len(out) >= 2:
                break
            else:
                continue
    print(''.join(out))