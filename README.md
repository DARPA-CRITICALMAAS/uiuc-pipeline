# CriticalMAAS Pipeline
This is the internal UIUC git repository for the DARPA CMAAS inference pipeline. The pipeline is designed to be run on hydro.

## Quickstart

<details>
<summary> Installing </summary>

  To get started with this pipeline you'll need to clone the repository and and install [requirements.txt](https://git.ncsa.illinois.edu/criticalmaas/pipeline/-/blob/abode_pipeline/requirements.txt). We recommend using python venv here to keep the working enviroment clean.

  ```bash
  # If your on hydro you'll need to load the python and cuda module
  #module load python/3.9.13 cuda/11.7.0 

  git clone git@git.ncsa.illinois.edu:criticalmaas/pipeline.git
  cd pipeline
  python3 -m venv venv
  source ./venv/bin/activate
  pip install -r requirements.txt
  ```

  This repository also makes use of submodules which will need to be initalized.

  ```bash
  git submodule init
  git submodule update
  ```

</details>

<details>
<summary> Understanding Pipeline Inputs </summary>

  To perform inference with our pipeline only one data input is actually required and that is the map that you want to perform inference on. But there are a also other data inputs we can use to speed up and perform optional steps with. Each of these optional inputs needs to be structured so that the name is consistant with the input map. E.g. if you have `CA_Sage.tif` the legend will need be named `CA_Sage.json`

  This is visualization of what that structure looks like.
  ```bash
  data
  ├── Map_1.tif
  ├── Map_2.tif
  ├── ...
  └── Map_N.tif

  legends # Optional
  ├── Map_1.json
  ├── Map_2.json
  ├── ...
  └── Map_N.json

  layouts # Optional
  ├── Map_1.json
  ├── Map_2.json
  ├── ...
  └── Map_N.json

  validation # Optional
  ├── Map_1_lgd_1_poly.tif
  ├── Map_1_lgd_2_poly.tif
  ├── ...
  ├── Map_1_lgd_N_poly.tif
  ├── ...
  ├── Map_N_lgd_1_poly.tif
  ├── Map_N_lgd_2_poly.tif
  ├── ...
  └── MapN_lgdN_poly.tif
  ```
  It's also important to note that if you specify --legends and there is no corrosponding legend for a map file that is completly fine. Pipeline will just fallback to generating a legend for that specfic map. Same is true for layouts and validation.

</details>

<details>
<summary> Performing Inference </summary>

  To perform inference with one of our models we will need to run pipeline.py. Pipeline.py has 3 core required arguments to run :

  * --model  : The model to use for inference.
  * --data   : Directory containing data to perform inference on.
  * --output : Directory to save the output data of the pipeline to.

  The list of available models can be found [below](#available-models) with the release-tag being what you want to use for the argument.

  Note* You must have a gpu available to run pipeline

  ```bash
  # Example call to pipeline.py
  python pipeline.py --model "primordal-positron" --data mydata/images/ --output mydata/output/
  ```
  Running this will have "primordal-positron" run inference on every `.tif` file in the directory specifed by `--data`. The output rasters of this inference will then be saved as `.tif`s to the directory specifed by `--output` along with a geopackage file for each map. The geopackage file contains vector data for each legend item in the map. Output is saved as the pipeline runs so even if the pipeline were to crash in the middle of running, all maps that ran before the crash will have been saved.

  By default the pipeline will save logging information to `logs/Latest.log` this can be useful if you have any problems or want to see a detailed view of what the pipeline is doing. You can also change the log file location with `--log`

  For the further documentation on all the pipeline options see [below](#pipeline-parameters).

</details>

<details>
<summary> Running on Hydro </summary>

  For running the pipeline on hydro there are two options, you can manually run the pipeline with an interactive srun session or we can submit an automatic job using sbatch. I won't cover manually running with srun here as you can see how to do that in the [hydro docs](https://docs.ncsa.illinois.edu/systems/hydro/en/latest/user-guide/running-jobs.html#srun)
  but you will need to make sure to srun with `--partition=a100` flag as these are the only nodes with gpus on hydro.

  For running with sbatch we have two scripts `submit.sh` and `start_pipeline.sh`. When we run `submit.sh` that script will automatically start `start_pipeline.sh` on an a100 node. 

  start_pipeline.sh is where we will want to set the pipeline parameters for our run first, then once we are ready to run all we have to do is call
  ```bash
  sbatch submit.sh
  ```
  And that will start the job. We can view our pipelines progess by looking at `logs/job_%yourjobid%.log`. The slurm logs can also be found at `logs/slurm/%yourjobid%.e` if you have any errors.

  *Hint `tail -f logs/job_%yourjobid%.log` can be very useful for viewing these logs.
  You can also use `nvitop` when on the node that is running the job to view gpu statistics in realtime.

  **Please note that our job script assumes that you are using venv to setup your enviroment, if you are using another python enviroment manager E.g. Conda or virtualenvwrapper you will need to adapt the start_pipeline.sh script to your setup.*

</details>

<details>
<summary> Understanding Pipeline Outputs </summary>

  Pipeline can produce quite a few output files so it can be important to understand what each is. The key argument here is `--feedback` as that controls wether pipeline will output files that are intended for debugging its accuracy. When feedback is enabled the pipeline will save any legend data that was generated by the pipeline, create a visualization image for each legend analized in the validation step and save the validation score csv for each individual map. This results in the following output structure.

  ```bash
  output
  ├── %data%_scores.csv # If validation was enabled and feedback wasn't
  ├── Map_1_lgd_1_poly.tif
  ├── Map_1_lgd_2_poly.tif
  ├── ...
  ├── Map_1_lgd_N_poly.tif
  ├── ...
  ├── Map_N_lgd_1_poly.tif
  ├── Map_N_lgd_2_poly.tif
  ├── ...
  └── Map_N_lgd_N_poly.tif

  feedback
  ├── %data%_scores.csv # If validation was enabled
  ├── Map_1
  │   ├── Map_1.json # If a map legend was generated by pipeline
  │   ├── Map_1_Scores.csv         # If validation was enabled
  │   ├── val_map_1_lgd_1_poly.tif # ''
  │   ├── val_map_1_lgd_2_poly.tif # ''
  │   ├── ...                      # ''
  │   └── val_map_1_lgd_N_poly.tif # ''
  ├── ...
  └── Map_N
      ├── Map_N.json # If a map legend was generated by pipeline
      ├── Map_N_Scores.csv         # If validation was enabled
      ├── val_map_N_lgd_1_poly.tif # ''
      ├── val_map_N_lgd_2_poly.tif # ''
      ├── ...                      # ''
      └── val_map_N_lgd_N_poly.tif # ''
  ```

  Note that if feedback isn't turned on and validation is pipeline will still save all the scores in the output directory to `#%data%_results.csv`

</details>

## FAQ
Q. Where is data on hydro?

A. `/projects/bbym/shared/data`

Q. I've updated to the latest pipeline commit and it doesn't work

A. New requirements could have been added or submodules could have been updated. It's always a good idea to run 
```bash
pip install -r requirements.txt
git submodule init
git submodule update
```
if you are having issues after updating to the most recent commit.

Q. I'm having a issue with the pipeline that i couldn't find help for on this page

A. Should probably have a location to report bugs to IDK where yet do i have to do everything? :)

## Documentation

### Pipeline Parameters

* --config : optional ## Not implemented yet ##<br>
    The config file to use for the pipeline. Not implemented yet
* --log : optional<br>
    Option to set the file that pipeline logging will write to. Defaults to "logs/Latest.log".
* --model : required ## Not implemented yet ##<br>
    The release-tag of the model checkpoint that will be used to perform inference. Available release-tags for models are listed below.
    ***Currently --model will only run "primordial-positron", stil need to include to run though**
* --data : required<br>
    Directory containing the data to perform inference on. The program will run inference on any `.tif` files in this directory.
* --legends : optional<br>
    Directory containing precomputed legend data USGS json format. If option is provided the pipeline will use the precomputed legend data instead of generating its own. Filenames are expected to match their corrasponding map filename. E.g. a file named `data/CA_Sage.tif` would have a `legends/CA_Sage.json` file. Can increase Pipeline Performance by skipping legend extraction step.
* --layouts : optional<br>
    Directory containing precomputed map layout data Uncharted json format. If option is provided pipeline will use the layout to assist in legend extraction and inferencing. Filenames are expected to match their corrasponding map filename. E.g. a file named `data/CA_Sage.tif` would have a `layouts/CA_Sage.json` file. Can signifgantly increase the performance of the pipeline.
* --validation : optional<br>
    Directory containing the true raster segmentations. If option is provided the pipeline will perform the validation step (Scoring the results of predictions) with this data. Filenames are expected to match their corrasponding map filename and legend. E.g. if there is a legend for map CA_Sage called Mbv_poly the validation directory would have a `validation/CA_Sage_Mbv_poly.tif` file.
* --output : required<br>
    Directory to save the outputs of inference to. These output currently include the predicted raster for each legend item of each map and geopackage file for each map which contains all of the layer in vector format. If the directory doesn't exist it will be created. ***Geopackage saving is disabled as there is a bug when saving currently**
* --feedback : optional<br>
    Directory to save feedback on the pipeline to. If option is provided pipeline will save any legend data that was generated by the pipeline, visualization images of the validation step and validation score csvs for each map. If the directory doesn't exist it will be created. This option will incur a slight performance hit on the pipeline.

### Pipeline Config # Not implemented yet 
Pipeline config will likely contain model specific config options. Some ones planed are below.
* Patch_size
* Patch_overlap
* Batch_size?

### Available Models
<details>
<summary> Primordial-Positron </summary>

Git Repository : https://git.ncsa.illinois.edu/nj7/darpa_proj<br>
Lead Developer : Nathan<br>
Description : Attention U-net model<br>

#### Release Tags : 
* primordial-positron_0.0.3

</details>

<details>
<summary> Golden-Muscat (Not implemented in pipeline yet)</summary>

Git Repository : https://github.com/xiyuez2/Darpa_Unet_Release <br>
Lead Developer : Xiyue<br>
Description : Unet model<br>

#### Release Tags :
* golden-muscat_0.0.1

</details>

<details>
<summary> Quantum-Sugar (Not implemented in pipeline yet)</summary>

Git Repository : https://github.com/Dongjiahua/DARPA_torch <br>
Lead Developer : Jiahua<br>
Description :<br>

#### Release Tags :
* quantum-sugar_0.0.1
* [quantum-sugar_0.0.2](https://github.com/Dongjiahua/DARPA_torch/releases/download/quantum-sugar_0.0.2/checkpoint.ckpt)

</details>