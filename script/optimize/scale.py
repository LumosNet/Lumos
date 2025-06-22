from PIL import Image
import os

def scale(directory, h, w):
    filenames = os.listdir(directory)
    with open("./build/path.txt", "w+") as fp:
        for name in filenames:
            if "png" in name:
                img = Image.open(directory+"/"+name)
                new = img.resize((h, w))
                new_p = "./build/data/" + name
                new.save(new_p)
                fp.write(new_p+"\n")
                
def WritePathes(pathes):
    fp_path = open(pathes, "r")
    with open("./build/path.txt", "w+") as fp:
        for path in fp_path.readlines():
            name = path.split('/')[-1]
            new = "./build/data/"+name
            fp.write(new)

if __name__ == "__main__":
    scale("/home/lumos/Lumos/data/dogvscat/train", 224, 224)
    WritePathes("/home/lumos/Lumos/data/dogvscat/train.txt")
