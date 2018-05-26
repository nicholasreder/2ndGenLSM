import argparse
import tables as tb
from skimage.external import tifffile
from skimage import restoration
import os.path
import h5py as h5
import numpy as np
from tqdm import tqdm
from PIL import ImageFilter
from PIL import Image
import cv2 as cv
import sys
import timeit
from joblib import Parallel, delayed
import multiprocessing as mp
import os
from functools import partial

def save_images(dir_name, filename_path, image_temp):
    image_temp = image_temp.transpose((1,0))
    image_temp = cv.resize(image_temp,(image_temp.shape[1], int(image_temp.shape[0]*1.414)), interpolation = cv.INTER_CUBIC)
    return image_temp

def main():

    tqdm.monitor_interval = 0
    start = timeit.default_timer()

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help='imaris file')
    parser.add_argument("--eosinMean", type = float, default = 1000, help = 'mean channel intensity')
    parser.add_argument("--nuclearMean", type = float, default = 1000, help = 'mean channel intensity')
    parser.add_argument("--zPlane", type = int, default = 500, help = 'z-plane index')
    parser.add_argument("--bkgLevel", type = float, default = 100, help = 'z-plane index')
    args = parser.parse_args()
    eosinMean = args.eosinMean
    nuclearMean = args.nuclearMean
    zPlane = args.zPlane
    bkgLevel = args.bkgLevel
    
    file_path_0 = args.filename
    filename_path = file_path_0.replace('.ims','')
    file_path = os.path.abspath(file_path_0)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    dir_name = os.path.dirname(file_path)
    dir_name_1 = dir_name + '\\' + filename_path
    dir_name_2 = dir_name + '\\' + filename_path + '\\0' 
    dir_name_3 = dir_name + '\\' + filename_path + '\\1'

    if not os.path.exists(dir_name_1):
        os.makedirs(dir_name_1)
    if not os.path.exists(dir_name_2):
        os.makedirs(dir_name_2)
    if not os.path.exists(dir_name_3):
        os.makedirs(dir_name_3)

    num_cores = mp.cpu_count()

    f = h5.File(file_path,'r')

    channel_0 = f['/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data']

    ymid = int(round(channel_0.shape[2]/2))
    yind = int(round(1000/1.414))

    temp_0 = channel_0[zPlane-2000:zPlane+2000,:,ymid-1000:ymid+1000]

    temp_0 = temp_0.astype('uint16')

    pool = mp.Pool(processes = num_cores)
    func_p = partial(save_images, dir_name, filename_path)
    img_temp = pool.map(func_p, (temp_0[:,z,:] for z in xrange(0, temp_0.shape[1])))

    for z in xrange(0, temp_0.shape[1]):
        tifffile.imsave(dir_name + '\\' + filename_path + '\\' + str(0) + '\\' + str(z).zfill(6) + '.tif', img_temp[z])
    
    pool.close()

    channel_0 = f['/DataSet/ResolutionLevel 0/TimePoint 0/Channel 1/Data']

    ymid = int(round(channel_0.shape[2]/2))
    yind = int(round(1000/1.414))

    temp_0 = channel_0[zPlane-2000:zPlane+2000,:,ymid-1000:ymid+1000]

    temp_0 = temp_0.astype('uint16')

    pool = mp.Pool(processes = num_cores)
    func_p = partial(save_images, dir_name, filename_path)
    img_temp = pool.map(func_p, (temp_0[:,z,:] for z in xrange(0, temp_0.shape[1])))

    for z in xrange(0, temp_0.shape[1]):
        tifffile.imsave(dir_name + '\\' + filename_path + '\\' + str(1) + '\\' + str(z).zfill(6) + '.tif', img_temp[z])
    
    pool.close()

    f.close()

    stop = timeit.default_timer()
    print('Total time: {0}s'.format(stop-start))

if __name__ == '__main__':
    print("Converting en-face plane from Imaris file to single RGB H&E TIFF file...")
    main()
