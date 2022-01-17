import os
import operator

path = '/home/btboay/darknet/CCTSDB/CCTSDB/GroundTruth/groundtruth0000-9999.txt'
root = '/home/btboay/darknet/CCTSDB/CCTSDB'

with open(path, 'r') as f:
    lines = f.readlines()
    n = 0
    for line in lines:
        print(n)
        l = line.split(';')
        num = int(l[0].split('.')[0])
        file = ''
        if num >= 0 and num < 1000:
            file = '{}{}{}'.format(root+'/', 'image0000-0999/', l[0].split('.')[0]+'.txt')
        if num >= 1000 and num < 2000:
            file = '{}{}{}'.format(root+'/', 'image1000-1999/', l[0].split('.')[0]+'.txt')
        if num >= 2000 and num < 3000:
            file = '{}{}{}'.format(root+'/', 'image2000-2999/', l[0].split('.')[0]+'.txt')
        if num >= 3000 and num < 4000:
            file = '{}{}{}'.format(root+'/', 'image3000-3999/', l[0].split('.')[0]+'.txt')
        if num >= 4000 and num < 5000:
            file = '{}{}{}'.format(root+'/', 'image4000-4999/', l[0].split('.')[0]+'.txt')
        if num >= 5000 and num < 6000:
            file = '{}{}{}'.format(root+'/', 'image5000-5999/', l[0].split('.')[0]+'.txt')
        if num >= 6000 and num < 7000:
            file = '{}{}{}'.format(root+'/', 'image6000-6999/', l[0].split('.')[0]+'.txt')
        if num >= 7000 and num < 8000:
            file = '{}{}{}'.format(root+'/', 'image7000-7999/', l[0].split('.')[0]+'.txt')
        if num >= 8000 and num < 9000:
            file = '{}{}{}'.format(root+'/', 'image8000-8999/', l[0].split('.')[0]+'.txt')
        if num >= 9000 and num < 10000:
            file = '{}{}{}'.format(root+'/', 'image9000-9999/', l[0].split('.')[0]+'.txt')
        id = ''
        if operator.eq(str(l[-1]).strip(), 'warning'):
            id = '0'
        if operator.eq(str(l[-1]).strip(), 'prohibitory'):
            id = '1'
        if operator.eq(str(l[-1]).strip(), 'mandatory'):
            id = '2'
        width = float(l[3]) - float(l[1])
        height = float(l[4]) - float(l[2])
        c1 = float(l[1]) + width / 2
        c2 = float(l[2]) + height / 2
        data = '{}{}{}{}{}'.format(id+' ', str(width)+' ', str(height)+' ', str(c1)+' ', str(c2))
        print(data)
        with open(file, 'a') as fd:
            fd.write(data+'\n')
            fd.close()
        n += 1
    f.close()
# 原：左上角、右下角坐标  改为：长款和中心点坐标