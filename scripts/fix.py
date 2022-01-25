from fileinput import close
import os
import random
import numpy as np

path1 = "/home/btboay/lumos-matrix/mnist/data.txt"
path2 = "/home/btboay/lumos-matrix/mnist/data2.txt"

new_lines = []

with open(path1, 'r') as f:
    lines = f.readlines()
    index_l = [x for x in range(len(lines))]
    indexs = random.sample(range(len(index_l)), len(lines))
    l = np.asarray(index_l)[indexs]
    for i in l:
        new_lines.append(lines[l[i]])
    f.close()

with open(path2, 'a') as fd:
    for line in new_lines:
        fd.write(line)
    fd.close()