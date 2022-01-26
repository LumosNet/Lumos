import os
from posixpath import split
import random
import numpy as np

path1 = "/home/btboay/lumos-matrix/mnist/data.txt"
path2 = "/home/btboay/lumos-matrix/mnist/label.txt"

new_lines = []

with open(path1, 'r') as f:
    lines = f.readlines()
    for line in lines:
        res = line.split('.')
        nl = res[0] + '.' + res[1]
        nl += ".txt\n"
        new_lines.append(nl)
    f.close()

with open(path2, 'a') as fd:
    for line in new_lines:
        fd.write(line)
    fd.close()