import os

def get_path(path, data_path, label_path):
    f = open(data_path, 'w')
    p = open(label_path, "w")
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.png' in file:
                f.write("{}{}{}\n".format(root, '/', file))
                txt = root + '/' + file.split('.')[0] + '.txt'
                p.write("{}\n".format(txt))
                txtf = open(txt, 'w')
                txtf.write("{}\n".format(root[-1]))
                txtf.close()
    f.close()
    p.close()

def delete_txt(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.txt' in file and 'groundtruth' not in file:
                os.remove("{}{}{}".format(root, '/', file))

if __name__ == '__main__':
    get_path('/home/btboay/MNIST/training', '/home/btboay/lumos-matrix/mnist/data.txt', '/home/btboay/lumos-matrix/mnist/label.txt')
    # delete_txt('/home/btboay/MNIST/training')