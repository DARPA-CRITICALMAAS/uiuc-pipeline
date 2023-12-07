import boto3
import glob
import os
import logging

class S3:
    def __init__(self, access_key, secret_key, server, bucketname):
        self.s3_access_key = access_key
        self.s3_secret_key = secret_key
        self.s3_server = server
        self.s3_bucketname = bucketname

        self.logger = logging.getLogger('pipeline.s3')
        self.s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, endpoint_url=server)
        self.bucket = self.s3.Bucket(self.s3_bucketname)

    def list(self, prefix=''):
        return [o.key for o in self.bucket.objects.filter(Prefix=prefix)]

    def upload(self, filename, regex=False):
        outputs = []
        if regex:
            for file in glob.glob(filename):
                self.logger.debug(f"Uploading {file}")
                self.bucket.upload_file(file, file)
                outputs.append(file)
        else:
            self.logger.debug(f"Uploading {filename}")
            self.bucket.upload_file(filename, filename)
            outputs.append(filename)
        return outputs

    def download(self, filename, regex=False, folder='.'):
        outputs = []
        if regex:
            for file in self.list(filename):
                output = os.path.join(folder, file)
                outputs.append(output)
                if not os.path.exists(output):
                    self.logger.debug(f"Downloading {file}")
                    if not os.path.exists(os.path.dirname(output)):
                        os.makedirs(os.path.dirname(output))
                    self.bucket.download_file(file, output)
        else:
            output = os.path.join(folder, filename)
            outputs.append(output)
            if not os.path.exists(output):
                self.logger.debug(f"Downloading {filename}")
                if not os.path.exists(os.path.dirname(output)):
                    os.makedirs(os.path.dirname(output))
                self.bucket.download_file(filename, output)
        return outputs
