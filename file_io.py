import numpy as np
import os
import boto3
from io import BytesIO
import skimage.external.tifffile as tif
import tempfile
from PIL import Image


def array_to_s3(arr, s3_fname ,cci):
    """
    Place an array directly into an S3 tiff file
    
    Parameters
    ----------
    arr : 2D numpy array/memmap
    
    s3_fname : str
        Full path to the S3 object to create (e.g., '/test/test.tif'
    
    cci : cottoncandy interface object    
    """
    # Create a PIL image object:
    im = Image.fromarray(arr)
    # This stands in place of the file to save to:
    output = BytesIO()
    # Do the saving: 
    im.save(output, format="tiff")
    # Need to seek back to the beginning of the file:
    output.seek(0)
    # This will create the object if it doesn't exist:
    obj = cci.get_object(s3_fname)
    # Off we go:
    obj.upload_fileobj(output)

    
def s3_to_array(f, cci):
    """ 
    Read a tif file straight from an S3 bucket (provided as a cottoncandy 
    interface) into a numpy array
   
    Parameters
    ----------
    f : str
        The name of the file in the bucket
    
    cci : cottoncandy interface
    
    
    """
    o = cci.download_object(f)
    b = BytesIO(o)
    t = tif.TiffFile(b)
    a = t.asarray()
    return a


def read_strip_files(file_list, files_per_strip, ss, cci, dtype, shape):
    """
    From a given list of files read all the tifs in one strip
    and return a memory-mapped array with the data.
    
    Parameters
    ----------
    file_list : list 
        All the file names from one experiment, ordered 
        according to strips 
    
    files_per_strip : int
        How many files (sheets) in each strip.
    
    ss : int
        A strip index.
    
    cci : a cottoncandy interface
    
    Return 
    ------
    Memory-mapped array with dimensions (z, width, sheets)
    """
    mm_fd, mm_fname = tempfile.mkstemp(suffix='.memmap')    
    strip_mm = np.memmap(mm_fname, dtype=dtype, 
                         shape=(files_per_strip, shape[0], shape[1]))
    for ii in range(files_per_strip):
        image_file = file_list[ss * files_per_strip + ii]
        strip_mm[ii] = s3_to_array(image_file, cci)
    
    mm_roll = np.swapaxes(np.swapaxes(strip_mm, 0, 1), 1, 2)
    # Strips are rastered back and forth, so we flip the odds
    if np.mod(ss, 2):
        return mm_roll[..., ::-1]
    return mm_roll



def download_s3(remote_fname, local_fname, bucket_name="alpenglowoptics"):
    """
    Download a file from S3 to our local file-system
    """
    if not os.path.exists(local_fname):
        s3 = boto3.resource('s3')
        b = s3.Bucket(bucket_name)
        b.download_file(remote_fname, local_fname)    

def create_zstack(scan_name, strip_num, z_levels, sample_image_num = 55):
    """
    creates a zstack from a series of tiff frames for a single strip
    
    Parameters
    ----------
    scan_name = the label for the scan files (string)
    strip_num = the index of the strip (int)
    z_levels = number of pixels in height dimension (int)
    sample_image_num = arbitrarily chosen frame for reading dtype and shape for each frame (int)
    
    Returns
    -------
    ImageCollection
    """
  
    #downloading frames from s3 to ec2 instance
    for x in range(1, z_levels+1):
        fname = "%06d_%06d.tif" % (strip_num, x) # XXX we need to enforce 06d, start from 0
        download_s3('%s/%06d/' %(scan_name, strip_num) + fname, '../data/%s/%06d/' %(scan_name, strip_num) + fname)    

    imread = delayed(skimage.io.imread, pure=True)  # Lazy version of imread
    filenames = []
    for x in range(0, z_levels):
        fname = "%06d_%06d.tif" % (strip_num, x)
        filenames.append('../data/%s/%06d/' %(scan_name, strip_num) + fname)

    lazy_values = [imread(filename) for filename in filenames]    
    sample = skimage.io.imread(filenames[sample_image_num])
    arrays = [da.from_delayed(lazy_value,           # Construct a small Dask array
                              dtype=sample.dtype,   # for every lazy value
                              shape=sample.shape)
              for lazy_value in lazy_values]

    stack = da.stack(arrays, axis=0)  

    for z in range(1,stack.shape[1]):
        filename = 'zstack_%06d_%06d.tif' % (strip_num, z) 
        tiff.imsave(filename, stack[:,z,:].compute())
    zstack = ImageCollection('zstack_%06d_*.tif' % strip_num )
    return zstack