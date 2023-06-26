import os
import sys
import shutil

def myMakedirs(path, overwrite):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' create directory successfully!')
    else:
        if overwrite:
            shutil.rmtree(path)
            os.makedirs(path)
            print(path, ' overwrite!')
        else:
            print(path, ' is already existed!')
            sys.exit()

def getSubFiles(pth, type):
    pth_list = []
    for s in os.listdir(pth):
        if type in s:
            pth_list.append(os.path.join(pth, s))
    return pth_list

def read_data_list(path, encoding=None):
    train_file = open(path, 'r', encoding=encoding)
    train_list = train_file.readlines()
    train_file.close()
    train_list = [p.replace('\n', '') for p in train_list]
    return train_list