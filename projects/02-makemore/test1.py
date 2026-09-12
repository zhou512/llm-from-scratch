""" import torch
import matplotlib
from pathlib import Path
input_path = Path("names.txt")
names = input_path.read_text().splitlines()

chars = sorted(set("".join(names)))
ctoi = {c:i+1 for i,c in enumerate(chars)}
print(ctoi) """

context = [0] * 3
print(type(context))
print(context)