import matplotlib.pyplot as plt
%matplotlib inline
from skimage.external.tifffile import imread
import skimage.external.tifffile as tif
import glob
cwd = os.getcwd()
path1 = cwd + '/*1.tif'
files1 = glob.glob(path1)
path2 = cwd + '/*2.tif'
files2 = glob.glob(path2)
beta2 = 0.05;
beta4 = 1.00;
beta6 = 0.544;
beta1 = 0.65;
beta3 = 0.85;
beta5 = 0.35;
for i in range(0,len(files1)):
    file1=files1[i]
    file2=files2[i]
    temp_0 = tif.imread(file1)
    temp_1 = tif.imread(file2)
    temp_0 = temp_0.astype(float)
    temp_1 = temp_1.astype(float)
    temp_0 = temp_0-50
    temp_1 = temp_1-50
    ind = np.where(temp_0 < 0)
    temp_0[ind] = 0
    ind = np.where(temp_1 < 0)
    temp_1[ind] = 0
    im = np.zeros((temp_0.shape[0], temp_0.shape[1], 3))
    im[:,:,0] = np.multiply(np.exp(-temp_0*beta1/1200), np.exp(-temp_1*beta2/1200))
    im[:,:,1] = np.multiply(np.exp(-temp_0*beta3/1200), np.exp(-temp_1*beta4/1200))
    im[:,:,2] = np.multiply(np.exp(-temp_0*beta5/1200), np.exp(-temp_1*beta6/1200))
    im = im*255
    im = im.astype('uint8')
    oldname = file1
    root = oldname.split('CH000001')[0]
    newname = root + 'rgb.tif'
    tifffile.imsave(newname, im)
