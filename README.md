# setup

This repository uses submodules. This requires a little bit of extra work to get started.

```bash
git clone git@git.ncsa.illinois.edu:criticalmaas/pipeline.git
cd pipeline
git submodule init
git submodule update
```

If new changes are made to the submodules, you will need to rerun:

```bash
git submodule update
```

On hydro make sure to load the modules for python and cuda:

```bash
module load python/3.9.13 cuda/12.2.1
```

Next you need to install requirements

```bash
python3 -m venv venv
pip install -r requirements.txt
```

**HACK HACK HACK**
You will need to copy the primordial-positron-pipeline.py to primordial-positron/pipeline.py

Setup the config.yaml file, and run `python pipeline.py`
```yaml
s3:
  access_key: 'XXXXXXXXXXXXXXXX'
  secret_key: 'YYYYYYYYYYYYYYYY'
  server: 'https://s3.example.com'
  bucketname: 'maps'

models:
  - name: primordial-positron
    folder: primordial-positron
    module: pipeline
    kwargs:
      featureType: Polygon
    checkpoint: primordial-positron/inference_model/Unet-attentionUnet.h5
```

## Pipeline Flow

- [x] load data from S3 (image + json) to input folder
- [x] load image to array
- [x] load legends to array {name=image}
- [x] load models (only tested with primordial-positron)
- [ ] run model (model starts but does not predict, needs above hack)
- [ ] save outputs (not implemented)
- [ ] save outputs to S3 (not implmeneted)

## TODO
- [ ] Allow user to input maps to process on command line
- [ ] Add server that launches jobs on HPC

# OLD NOTES

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

## Release : primordial-positron
### Lead Developer : Nathan

**Tag : primordial-positron_0.0.3**

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

## Release : golden-muscat
### Lead Developer : Xiyue

**Tag : golden-muscat_0.0.1**

model = git lfs
URL = https://github.com/xiyuez2/Darpa_Unet_Release

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

## Release : quantum-sugar
### Lead Developer : Jiahua

**Tag : quantum-sugar_0.0.1**

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
