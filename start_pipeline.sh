#!/bin/bash
echo "Job $SLURM_JOB_ID running on $(hostname)"

# Env Setup
module load python/3.9.13 cuda/11.7.0
source venv/bin/activate

### Pipeline parameters
python pipeline.py \
    -c garbage.yaml \
    --model "primordal-positron" \
    --log "logs/Job_$SLURM_JOB_ID.log"\
    --data testdata/images/ \
    --output testdata/output/ \
    --legends ../data/validation/usgs_legends/ \
    --image_layout ../data/validation/uncharted_masks/ \
    --validation ../data/validation/usgs_segmentations/

echo "Job terminating successfully"