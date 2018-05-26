import argparse
def s3_to_ec2(remote_fname, local_fname, bucket_name="raw-alpenglow/data/"):
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_fname", help = 'filename on s3')
    parser.add_argument("local_fname", help = 'intended filename on ec2')
    parser.add_argument("bucket_name", help = 'name of s3 bucket, including subfolder')
    args = parser.parse_args()
    remote_fname = args.remote_fname
    local_fname = args.local_fname
    bucket_name = args.bucket_name
    """
    Download a file from S3 to our local file-system
    """
    if not os.path.exists(local_fname):
        s3 = boto3.resource('s3')
        b = s3.Bucket(bucket_name)
        b.download_file(remote_fname, local_fname) 