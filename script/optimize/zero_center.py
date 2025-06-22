from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
import pickle

def zero_center(path):
    filenames = os.listdir(path)
    for name in filenames:
        if ".png" in name:
            img = Image.open(path + '/' + name)
            r,g,b = img.split()
            r = np.array(r)
            g = np.array(g)
            b = np.array(b)
            img_data = [r, g, b]
            data = []
            for i in img_data:
                for j in range(np.array(img).shape[0]):
                    data += list(i[j])
            mean = sum(data) / len(data)
            data = [(data[i]-mean)/255 for i in range(len(data))]
            with open('./build/new/'+name.split('.')[0], 'wb') as f:
                pickle.dump(data, f)
                f.close()

def new_path(file):
    with open(file, "r") as fp:
        with open("./build/path.txt", "w+") as fl:
            for line in fp.readlines():
                path = "./build/new/" + line.split('/')[-1].split('.')[0]
                fl.write(path+'\n')
            fl.close()
        fp.close()

zero_center("./build/data")
new_path("./build/train.txt")

