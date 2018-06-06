import os
import boto3

for i in range(0, 50991):
  fname_topro = ('X000350' + '_Y044520' + '_Z' + str(i*3).zfill(6) + '_CH' + str(1).zfill(6) + '.tif')

if not os.path.exists(fname_topro):
  s3 = boto3.resource('s3')
  b = s3.Bucket("raw-alpenglow/data/17-119_n2/")
  b.download_file(fname_topro, fname_topro)

for i in range(0, 50991):
  fname_topro = ('X000350' + '_Y044520' + '_Z' + str(i*3).zfill(6) + '_CH' + str(1).zfill(6) + '.tif')

if not os.path.exists(fname_topro):
  s3 = boto3.resource('s3')
  b = s3.Bucket("raw-alpenglow/data/17-119_n2/")
  b.download_file(fname_topro, fname_topro)
