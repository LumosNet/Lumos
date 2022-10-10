from math import e
import os
import glob
import math

def main():
    filename1 = glob.glob("obj/*.o")
    filename2 = glob.glob("./*.exe")
    filename3 = glob.glob("./lib/*.dll")
    filename4 = glob.glob("./lib/*.lib")
    filename5 = glob.glob("./*.o");
    filename6 = glob.glob("lumos/*.so")
    filename = filename1 + filename2 + filename3 + filename4 + filename5 + filename6
    for file in filename:
        os.remove(file)

if __name__ == "__main__":
    main()
