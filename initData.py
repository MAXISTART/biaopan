import cv2
import os
import glob
import numpy as np
# 读取路径下的每个图片的小分区，然后给每个folder下都创建一个txt文件，记录下label


# mat = cv2.imread(r'D:\VcProject\biaopan\data\goodImgs\1\1.jpg')
# cv2.imshow('aa', mat)
# order = cv2.waitKey()
# print(order)
readBasePath = r'D:\VcProject\biaopan\data\goodImgs'
dirs = os.listdir(readBasePath)
# 最后会会和所有之前的txt文件成为一个最终txt文件
lastOutputPath = r'D:\VcProject\biaopan\data\labels.txt'
startFolder = 460
sort_dirs = []
for fol in sorted([int(d) for d in dirs]):
    if fol >= startFolder:
        sort_dirs.append(str(fol))

for dir in sort_dirs:
    print('当前dir是： ', dir)
    # D:\VcProject\biaopan\data\goodImgs\1\
    spath = os.path.join(readBasePath, dir)
    # D:\VcProject\biaopan\data\goodImgs\1\1.jpg
    mserpaths = glob.glob(os.path.join(spath, '*.jpg'))
    maxsize = len(mserpaths)
    mserpaths = [os.path.join(spath, str(i) + '.jpg') for i in range(1, maxsize + 1)]
    mi = 0
    mserIndex = -1
    cv2.imshow('mser', np.zeros((36, 18), np.uint8))

    # 初始化要存储的数列
    labels = [-1 for i in range(0, maxsize+1)]
    while(True):
        order = cv2.waitKey()
        if order == 97:
            mserIndex = max(0, mserIndex-1)
            mat = cv2.imread(mserpaths[mserIndex])
            mat = cv2.resize(mat, (18, 36))
            cv2.imshow('mser', mat)
        elif order == 100:
            mserIndex = min(maxsize - 1, mserIndex+1)
            mat = cv2.imread(mserpaths[mserIndex])
            mat = cv2.resize(mat, (18, 36))
            cv2.imshow('mser', mat)
        elif order == 115:
            # 准备退出的时候就进行保存标记
            f = open(os.path.join(spath, 'label.txt'), 'a')
            f.writelines([mserpaths[i] + '===' + str(labels[i]) + '\n' for i in range(0, maxsize)])
            f.close()
            print('正在进入下一个文件-------------------')
            break
        elif order in range(48, 58):
            # 如果不是以上的指令，那就是要打标记
            label = order - 48
            labels[mserIndex] = label
            print('标记为---', label)
        else:
            labels[mserIndex] = -1
            print('标记为---', -1)


out = open(lastOutputPath, 'a')
# 最后汇合之前的label文件
for dir in sorted([int(d) for d in dirs]):
    spath = os.path.join(readBasePath, str(dir), 'label.txt')
    f = open(spath, 'r')
    out.writelines(f.readlines())
f.close()