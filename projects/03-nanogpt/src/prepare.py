import os
HERE = os.path.dirname(os.path.abspath(__file__))          # 本文件所在目录

with open(os.path.join(HERE,'..','data','input.txt'),"r",encoding='utf-8') as f:
    text_en = f.read()
print(f"text_en:{len(text_en)}")
with open(os.path.join(HERE,'..','data','input_zh.txt'),"r",encoding='utf-8') as f:
    text_zh = f.read()
print(f"text_zh:{len(text_zh)}")
chars_zh = sorted(list(set(text_zh)))
vocab_size_zh = len(chars_zh)
zhs = ' '.join(chars_zh)
print(zhs) 
