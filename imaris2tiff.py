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
    file_path = os.path.abspath(file_path_0)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    dir_name = os.path.dirname(file_path)

    f = h5.File(file_path,'r')

    channel_0 = f['/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data']
    channel_1 = f['/DataSet/ResolutionLevel 0/TimePoint 0/Channel 1/Data']
    
    beta2 = 0.05;
    beta4 = 1.00;
    beta6 = 0.544;

    beta1 = 0.65;
    beta3 = 0.85;
    beta5 = 0.35;

    temp_0 = channel_0[xrange(zPlane-1000,zPlane+1000),zPlane,:]
    temp_1 = channel_1[xrange(zPlane-1000,zPlane+1000),zPlane,:]
    temp_0 = temp_0.astype(float)
    temp_1 = temp_1.astype(float)

    # unsharp mask
    # temp_00 = cv.GaussianBlur(temp_0,(5,5),0)
    # temp_000 = cv.GaussianBlur(temp_00,(1,1),0)
    # alpha = 50
    # temp_0 = temp_00 + alpha*(temp_00 - temp_000)

    # temp_11 = cv.GaussianBlur(temp_1,(5,5),0)
    # temp_111 = cv.GaussianBlur(temp_11,(1,1),0)
    # alpha = 50
    # temp_1 = temp_11 + alpha*(temp_11 - temp_111)

    # unsharp mask 2
    # temp_00 = cv.GaussianBlur(temp_0,(21,21),4)
    # temp_000 = cv.addWeighted(temp_0, 1.5, temp_00, -0.5, 0, temp_0)
    # #idx = np.where(abs(temp_00 - temp_000) > 100)
    # #temp_0[idx] = temp_000[idx]
    # idx = np.where(temp_000 < 0)
    # temp_000[idx] = 0
    # temp_0 = temp_000

    # temp_11 = cv.GaussianBlur(temp_1,(21,21),4)
    # temp_111 = cv.addWeighted(temp_1, 1.5, temp_11, -0.5, 0, temp_1)
    # # idx = np.where(abs(temp_11 - temp_111) > 1000)
    # # temp_1[idx] = temp_111[idx]
    # idx = np.where(temp_111 < 0)
    # temp_111[idx] = 0
    # temp_1 = temp_111

    # PILLOW
    # temp_0 = Image.fromarray(temp_0,'F')
    # temp_1 = Image.fromarray(temp_1,'F')
    # temp_0 = temp_0.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=int(bkgLevel)))
    # temp_1 = temp_1.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=int(bkgLevel)))
    # temp_0 = np.asarray(temp_0)
    # temp_1 = np.asarray(temp_1)

    # deconvolution
    # psf = np.ones((5,5))/25
    # print(type(temp_0))
    # temp_0 = restoration.unsupervised_wiener(temp_0, psf)
    # temp_1 = restoration.unsupervised_wiener(temp_1, psf)
    # temp_0 = np.asarray(temp_0[0])
    # temp_1 = np.asarray(temp_1[0])

    temp_0 = temp_0-bkgLevel
    temp_1 = temp_1-bkgLevel
    ind = np.where(temp_0 < 0)
    temp_0[ind] = 0
    ind = np.where(temp_1 < 0)
    temp_1[ind] = 0
    #temp_0 = temp_0.transpose((1,0))
    #temp_1 = temp_1.transpose((1,0))
    #temp_0 = cv.resize(temp_0,(temp_0.shape[1], int(temp_0.shape[0]*1.414)), interpolation = cv.INTER_CUBIC)
    #temp_1 = cv.resize(temp_1,(temp_1.shape[1], int(temp_1.shape[0]*1.414)), interpolation = cv.INTER_CUBIC)
    #im = np.zeros((temp_0.shape[0], temp_0.shape[1], 3))
    #im[:,:,0] = np.multiply(np.exp(-temp_0*beta1/nuclearMean), np.exp(-temp_1*beta2/eosinMean))
    #im[:,:,1] = np.multiply(np.exp(-temp_0*beta3/nuclearMean), np.exp(-temp_1*beta4/eosinMean))
    #im[:,:,2] = np.multiply(np.exp(-temp_0*beta5/nuclearMean), np.exp(-temp_1*beta6/eosinMean))
    #im = im*255
    #im = im.astype('uint8')
    tifffile.imsave(dir_name + '\\' + file_name + '_Z' + str(zPlane).zfill(6) + '.tif', im)

    f.close()

    stop = timeit.default_timer()
    print('Total time: {0}s'.format(stop-start))

if __name__ == '__main__':
    print("Converting en-face plane from Imaris file to single RGB H&E TIFF file...")
    main()
