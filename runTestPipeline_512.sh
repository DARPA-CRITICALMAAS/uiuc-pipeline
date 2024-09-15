#TAG="0.4.2" #TAG="0.4.2"
TAG="pr-26"
IMAGE="criticalmaas-pipeline_${TAG}.sif"
# apptainer pull ${IMAGE} docker://ncsa/criticalmaas-pipeline:${TAG}
TEST_MODEL="seasoned_lynx"
# DEST_DIR="/projects/bbym/shared/testingPR/${TAG}/${TEST_MODEL}/validation_w_layout"
DEST_DIR="/projects/bcxi/shared/seasoned_lynx_testing/patch_512"
SRC_DATA="/projects/bcxi/shared/datasets/validation"
mkdir -p ${DEST_DIR}/feedback
mkdir -p ${DEST_DIR}/logs
mkdir -p ${DEST_DIR}/output

#srun -A bcxi-tgirails --time=4:00:00 --partition=gpu --ntasks-per-node=16 --gpus=4 --mem=50g --nodes=1 --pty /bin/bash

# apptainer run --nv -B ${SRC_DATA}:/data -B ${DEST_DIR}/feedback:/feedback \
#           -B ${DEST_DIR}/logs:/logs -B ${DEST_DIR}/output:/output ${IMAGE} \

python pipeline.py \
    -v --data ${SRC_DATA}/images \
	--legends ${SRC_DATA}/legends --output ${DEST_DIR}/output  \
	--layouts ${SRC_DATA}/layouts \
    --feedback ${DEST_DIR}/feedback \
    --log ${DEST_DIR}/logs/log.log --model ${TEST_MODEL} \
	--validation ${SRC_DATA}/true_segmentations \
    --output_types raster_masks \
    --patch_size 512


# make sure to change the patch size to the desired size