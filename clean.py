import os
import glob

def main():
    filename1 = glob.glob("obj/*.o")
    filename2 = glob.glob("./*.exe")
    filename3 = glob.glob("./lib/*.dll")
    filename4 = glob.glob("./lib/*.lib")
    filename = filename1 + filename2 + filename3 + filename4
    for file in filename:
        os.remove(file)

if __name__ == "__main__":
    main()