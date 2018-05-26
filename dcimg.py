import argparse
from shutil import copyfile
import numpy as np, ctypes
import cv2
import os.path
import errno
import sys
import timeit
from PIL import Image
from skimage.external import tifffile
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial
from skimage.external.tifffile import TiffWriter

def get_mean_int(img):

	interval = np.round(img_3d.shape[1]/100)
	interval = interval.astype(int)
	tileSize = 64
	bkgLevel = 200
	imgShape = [256,2048]
	rows = np.arange(0, imgShape[0]-1, tileSize)
	cols = np.arange(0, imgShape[1]-1, tileSize)
	frames = np.arange(interval, img_3d.shape[1]-interval, interval)
	M = np.zeros((len(rows)-1, len(cols)-1, len(frames)-1), dtype = float)
	meanInt = np.zeros(len(frames))

	idx = 0

	for k in frames:
		img = img_3d[:, k, :]
		img = img.astype(float)
		for i in range(1,len(rows)):
			for j in range(1,len(cols)):
				ROI = img[rows[i-1]:rows[i], cols[j-1]:cols[j]]
				fkg_ind = np.where(ROI > bkgLevel)
				if fkg_ind[0].size==0:
					Mtemp = np.mean(ROI)
				else:
					Mtemp = np.mean(ROI[fkg_ind[0]])
				M[i-1, j-1] = Mtemp + M[i-1, j-1]

		fkg_ind = np.where(M > bkgLevel)
		if fkg_ind[0].size==0:
			idx = idx
		else:
			meanInt[idx] = np.median(M[fkg_ind])
			idx = idx + 1

	meanInt = meanInt[0:idx]
	meanInt = round(np.median(meanInt))

	return meanInt

def normalize_images(meanInt, img):

	tileSize = 64
	bkgLevel = 500
	imgShape = [256,2048]
	rows = np.arange(0, imgShape[0]-1, tileSize)
	cols = np.arange(0, imgShape[1]-1, tileSize)
	M=np.zeros((len(rows)-1, len(cols)-1), dtype = float)

	img = img.astype(float)

	for i in range(1,len(rows)):
		for j in range(1,len(cols)):
			ROI = img[rows[i-1]:rows[i], cols[j-1]:cols[j]]
			fkg_ind = np.where(ROI > bkgLevel)
			if fkg_ind[0].size==0:
				Mtemp = np.mean(ROI)
			else:
				Mtemp = np.mean(ROI[fkg_ind[0]])
			M[i-1, j-1] = Mtemp + M[i-1, j-1]

	Msum = M/meanInt
	Msum = cv2.resize(Msum, (imgShape[1], imgShape[0]), interpolation = cv2.INTER_LINEAR)
	bkg_ind = np.where(Msum <= bkgLevel)
	img2 = np.zeros(Msum.shape)
	img2[bkg_ind] = 1
	img2 = cv2.resize(img2, (imgShape[1], imgShape[0]), interpolation = cv2.INTER_LINEAR)
	img4 = Msum + img2
	img3 = np.divide(img, img4)
	img = img3.astype('uint16')

	return img

if __name__ == '__main__':

	print("Pre-processing code")

	tqdm.monitor_interval = 0

	parser = argparse.ArgumentParser()
	parser.add_argument("filename", help='file name of .dcimg')
	parser.add_argument("--channel", type = int, default = 0, help ="color channel")
	parser.add_argument("--dir", choices=range(2), type = int, default = 0, help="direction")
	parser.add_argument("--compress", action = 'store_true', help = "compress hdf5 by B3D")
	parser.add_argument("--delete", action = 'store_true', help = 'delete the dcimg file')
	parser.add_argument("--meanInt", type = float, default = 1, help = 'mean channel intensity')
	args = parser.parse_args()

	direction = args.dir
	channel = args.channel
	meanInt = args.meanInt

	# path
	cwd = os.getcwd()
	file_path_0 = args.filename
	file_path = os.path.abspath(file_path_0)
	#file_path_utf8 = bytes(file_path,"utf8")
	lib_path='.\\DCIMG.dll'
	file_name = os.path.splitext(os.path.basename(file_path))[0]
	dir_name = os.path.dirname(file_path_0)
        slash=dir_name.find('\\')
        dir_name=dir_name[slash:]
        dir_name='DATA_' + str(channel) + '\\' + dir_name
	data_name = os.path.abspath(dir_name)
        data_name2 = data_name
        
	if args.compress:
		compression = True
	else:
		compression = False

	# check the nesassary files
	if os.path.exists(file_path):
		print('File found.')
	else:
		print('File not found.')
		sys.exit()
		
	if os.path.exists(lib_path):
		print('Dynamic-link library found.')
	else:
		print('Dynamic-link library not found.')
		sys.exit()
	# load the DLL to read the .DCIMG file
	libc = ctypes.cdll.LoadLibrary(lib_path)
	if libc.OpenDcimg(file_path):
		print('Dynamic-link library loaded.')
	else:
		print('Dynamic-link library not loaded.')
		sys.exit()

	# get the sessioin number of the dcimg file
	nTotalSession = libc.GetTotalSessionCount()
	session_index = 0

	nFrame = ctypes.c_int()
	width = ctypes.c_int()
	height = ctypes.c_int()
	rowbyte = ctypes.c_int()

	start = timeit.default_timer()

	if not libc.GetImageInformation(session_index,ctypes.byref(nFrame),
							 ctypes.byref(width),ctypes.byref(height),ctypes.byref(rowbyte)):
		print('Failed to get information about the image.')
		sys.exit()
	else:
		nFrame, width, height, rowbyte = nFrame.value, width.value, height.value,rowbyte.value
		# raw is the raw data of the image
		raw_type=(ctypes.c_ubyte*(rowbyte* height))
		raw = raw_type()
		
		print('Session #{0}: {1} frames.'.format(session_index, nFrame))
		print('Width x height: {0} x {1}.'.format(width, height))

		#read images
		print('Reading images...')
		img_3d = np.zeros((nFrame, height, width), dtype = 'uint16')
		for frame_index in tqdm(xrange(nFrame)): 
			if libc.AccessRawFrame(frame_index, session_index,ctypes.byref(raw)):
				print('Failed to load image from the .dcimg file.')
				sys.exit()
			else:
				img = np.ctypeslib.as_array(raw)
				img = np.fromstring(img, dtype = '<i2').reshape(height,width)
				img = abs(img - 100)
				img_3d[frame_index] = img

		stop = timeit.default_timer()
		print('Total time: {0}s'.format(stop-start))

		#shear images


		img_3d = img_3d.transpose((1,0,2))

		print('Shearing images...')

		start = timeit.default_timer()

		if not os.path.exists(data_name2):
			try:
				os.makedirs(data_name2)
			except OSError as exception:
				if exception.errno != errno.EEXIST or not os.path.isdir(path):
					raise					
		
		for i in tqdm(xrange(height)):
			if direction:
				img_3d[i,i:nFrame,:] = img_3d[i,height:img_3d.shape[1]-i,:]
				#img_3d[i,:,:] = img_3d[i,i:i+nFrame,:]

			else:
				img_3d[i,height-1-i:nFrame,:] = img_3d[i,0:img_3d.shape[1]-height+1+i,:]
				#img_3d[i,:,:] = img_3d[i, height-1-i:height-1-i+nFrame,:]

		if direction:
			img_3d = img_3d[:,0:nFrame-height,:]
		else:
			img_3d = img_3d[:,height:nFrame,:]

		img_3d = img_3d.transpose((1,0,2))
		img_3d = np.flip(img_3d,1)

		stop = timeit.default_timer()
		print('Total time: {0}s'.format(stop-start))

		#normalize images

		print('Normalizing images...')

		start = timeit.default_timer()

		pool = mp.Pool(processes = 4)
		func_p = partial(normalize_images, meanInt)
		img_temp = pool.map(func_p, (img_3d[z,:,:] for z in xrange(0, img_3d.shape[0])))
		pool.close()
		
		for z in xrange(0, img_3d.shape[0]):
			img_3d[z,:,:] = img_temp[z]

		stop = timeit.default_timer()
		print('Total time: {0}s'.format(stop-start))

		#write images
		
		print('Writing images...')

		start = timeit.default_timer()

		# for k in tqdm(xrange(img_3d.shape[0])):
		# 	#cv2.imwrite(data_name2+'\\'+file_name + '_' + str(k).zfill(6) +'.tif', img_3d[k,:,:])
		# 	tifffile.imsave(data_name2 + '\\' + str(k*3).zfill(6) + '.tiff', img_3d[k,:,:])

		with TiffWriter(data_name2+'\\'+file_name+'.tif', bigtiff =True) as tif:
			tif.save(img_3d)

		stop = timeit.default_timer()
		print('Total time: {0}s'.format(stop-start))

	libc.CloseDcimg()

	if args.delete:
		os.remove(file_path)
