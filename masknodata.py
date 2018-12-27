import rasterio as rio
from os import path
from glob import glob
import numpy as np
def train_set_list(src_folder, label_folder):
    srclist = glob(path.join(src_folder, '*.tif'))
    labellist = [path.join(label_folder, path.split(i)[-1].replace('src', 'label')) for i in srclist]
    return srclist, labellist
if __name__=='__main__':
    root_dir=r'E:\PROJECTS\lanmei_watershed\snap'
    GTSFoldername=path.join(root_dir,'GTS')
    srcFoldername=path.join(root_dir,'src')
    srclist, labellist=train_set_list(srcFoldername,GTSFoldername)
    for src_path,label_path in zip(srclist,labellist):
        GTS_img=rio.open(label_path).read().transpose(1,2,0)
        src_file=rio.open(src_path,'r+')
        src_img=src_file.read().transpose(1,2,0)
        src_img[np.all(GTS_img==np.array([0,0,0]),axis=2)]=[0,0,0,0,0,0]
        src_img=src_img.transpose(2,0,1)
        src_file.write(src_img)
