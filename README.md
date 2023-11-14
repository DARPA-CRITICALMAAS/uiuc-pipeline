# pipeline flow

INPUT = map URL + layer (can be all)

- download image
  - can copy from disk
  - should be geotiff
- legend extractor
  - generate json document, should be PageExtractor.json
- convert h5
  - can run in parallel with legend
  - skip if we use tiff images
- run model
  - use either tiff or h5
  - can run models in parallel
  - create geotiff
- upload image
  - use boto3 to upload to S3

# models

check gpu for tensorflow
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Nathan : primordial-positron 

URL = https://git.ncsa.illinois.edu/nj7/darpa_proj

module load python/3.9.13 cuda/12.2.1
python3 -m venv venv/primordial-positron

. ./venv/primordial-positron/bin/activate
pip install --upgrade pip

pip install ~/h5image/
pip install -r primordial-positron/requirements-inference.txt

cd primordial-positron
. ../venv/primordial-positron/bin/activate
mkdir -p ../output/primordial-positron
python inference.py --mapPath "/projects/bbym/shared/data/commonPatchData/256/OK_250K.hdf5" --featureType "Polygon" --modelPath ./inference_model/Unet-attentionUnet.h5 --outputPath "../output/primordial-positron"
deactivate
cd ..

## Xiyue : golden-muscat

model = git lfs
URL = https://github.com/xiyuez2/Darpa_Unet

cd
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.0/git-lfs-linux-amd64-v3.4.0.tar.gz
tar xf git-lfs-linux-amd64-v3.4.0.tar.gz

export PATH=${PATH}:${HOME}/git-lfs-3.4.0

module load python/3.9.13 cuda/12.2.1
python3 -m venv venv/golden-muscat
. ./venv/golden-muscat/bin/activate
pip install --upgrade pip
pip install -r golden-muscat/requirements-inference.txt


cd primordial-positron
. ../venv/primordial-positron/bin/activate
mkdir -p ../output/primordial-positron
python inference.py --mapPath "/projects/bbym/shared/data/commonPatchData/256/OK_250K.hdf5" --featureType "Polygon" --modelPath ./inference_model/Unet-attentionUnet.h5 --outputPath "../output/primordial-positron"
deactivate
cd ..

## Jiahua : quantum-sugar

model = https://github.com/Dongjiahua/DARPA_torch/releases/download/quantum-sugar_0.0.2/checkpoint.ckpt
URL = https://github.com/Dongjiahua/DARPA_torch
Works = PT

module load python/3.9.13 cuda/12.2.1
python3 -m venv venv/quantum-sugar
. ./venv/quantum-sugar/bin/activate
pip install --upgrade pip
pip install -r quantum-sugar/requirements-inference.txt

cd quantum-sugar
. ../venv/quantum-sugar/bin/activate
mkdir -p ../output/quantum-sugar
python inference.py --mapPath "/projects/bbym/shared/data/commonPatchData/validation/256/OR_Carlton.hdf5" --modelPath 'checkpoint.ckpt' --outputPath '../output/quantum-sugar'
deactivate
cd ..
