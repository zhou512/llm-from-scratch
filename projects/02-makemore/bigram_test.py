import torch
import matplotlib
from pathlib import Path
input_path = Path("names.txt")
names = input_path.read_text().splitlines()
spices = []
freqs = torch.zeros((27,27),dtype=torch.int32)
chars = sorted(set("".join(names)))
ctoi = {c:i+1 for i,c in enumerate(chars)}
ctoi["."] = 0
for name in names:
    tokens = ['<S>'] + list(name) + ['<E>']
    for a,b in zip(tokens,tokens[1:]):
        if a == '<S>':
            a = '.'
        if b == '<E>':
            b = '.'
        freqs[ctoi[a],ctoi[b]] += 1
    # spices.extend(a + b for a,b in zip(tokens,tokens[1:]))
#第i行第j列表示以第i个字符开头，下一个字是j的频次 freq[i,j]
itoc = {i:c for c,i in ctoi.items()}
print(itoc)
P = freqs.float()
P = P / P.sum(1,keepdim=True)
g = torch.Generator().manual_seed(123456)
for i in range(20):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        if ix == 0:  #生成出0代表结束
            if len(out) >= 3:
                break
            else:
                continue
        out.append(itoc[ix])
        if len(out) > 20:
            break
    print(''.join(out))
#print(spices[:10])

#建立一个二维张量，统计各个字段出现的次数

