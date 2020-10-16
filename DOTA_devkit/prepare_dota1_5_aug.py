import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool
from DOTA2COCO import DOTA2COCOTest, DOTA2COCOTrain
import argparse

wordname_16 = ['small ship',  'large ship', 'civilian aircraft', 'military aircraft', 'small car', 
               'bus', 'truck', 'train', 'crane', 'bridge', 'oil tank', 'dam', 'athletic field', 'helipad', 'roundabout']

def parse_args():
    parser = argparse.ArgumentParser(description='prepare dota1')
    parser.add_argument('--srcpath', default='/home/dingjian/project/dota')
    parser.add_argument('--dstpath', default=r'/home/dingjian/workfs/dota1-split-1024',
                        help='prepare data')
    args = parser.parse_args()

    return args

def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)
def filecopy(srcpath, dstpath, num_process=16):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)

def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)

def filemove(srcpath, dstpath, num_process=15):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)

def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = util.mybasename(file)
            f_out.write(basename + '\n')

def prepare(srcpath, dstpath):
    """
    :param srcpath: train, val, test
          train --> trainval1024, val --> trainval1024, test --> test1024
    :return:
    """
    if not os.path.exists(os.path.join(dstpath, 'test1024')):
        os.makedirs(os.path.join(dstpath, 'test1024'))
    if not os.path.exists(os.path.join(dstpath, 'test1024_ms')):
        os.makedirs(os.path.join(dstpath, 'test1024_ms'))
    if not os.path.exists(os.path.join(dstpath, 'trainval1024')):
        os.makedirs(os.path.join(dstpath, 'trainval1024'))
    if not os.path.exists(os.path.join(dstpath, 'trainval1024_ms')):
        os.makedirs(os.path.join(dstpath, 'trainval1024_ms'))

    split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train2017'),
                       os.path.join(dstpath, 'trainval1024'),
                      gap=500,
                      subsize=1024,
                      num_process=16
                      )
    split_train.splitdata(1)

    split_train_ms = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train2017'),
                        os.path.join(dstpath, 'trainval1024_ms'),
                        gap=500,
                        subsize=1024,
                        num_process=16)
    split_train_ms.splitdata(0.5)
    split_train_ms.splitdata(1.5)

    split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val2017'),
                       os.path.join(dstpath, 'trainval1024'),
                      gap=500,
                      subsize=1024,
                      num_process=16
                      )
    split_val.splitdata(1)

    split_val_ms = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val2017'),
                        os.path.join(dstpath, 'trainval1024_ms'),
                        gap=500,
                        subsize=1024,
                        num_process=16)
    split_val_ms.splitdata(0.5)
    split_val_ms.splitdata(1.5)

    split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath, 'test', 'images'),
                       os.path.join(dstpath, 'test1024', 'images'),
                      gap=500,
                      subsize=1024,
                      num_process=16
                      )
    split_test.splitdata(1)

    split_test_ms = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath, 'test', 'images'),
                       os.path.join(dstpath, 'test1024_ms', 'images'),
                      gap=500,
                      subsize=1024,
                      num_process=16
                      )
    split_test_ms.splitdata(0.5)
    split_test_ms.splitdata(1.5)

    DOTA2COCOTrain(os.path.join(dstpath, 'trainval1024'), os.path.join(dstpath, 'trainval1024', 'DOTA1_5_trainval1024.json'), wordname_16, difficult='2')
    DOTA2COCOTrain(os.path.join(dstpath, 'trainval1024_ms'), os.path.join(dstpath, 'trainval1024_ms', 'DOTA1_5_trainval1024_ms.json'), wordname_16, difficult='2')

    DOTA2COCOTest(os.path.join(dstpath, 'test1024'), os.path.join(dstpath, 'test1024', 'DOTA1_5_test1024.json'), wordname_16)
    DOTA2COCOTest(os.path.join(dstpath, 'test1024_ms'), os.path.join(dstpath, 'test1024_ms', 'DOTA1_5_test1024_ms.json'), wordname_16)
if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    prepare(srcpath, dstpath)
