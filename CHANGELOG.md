# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

## [0.4.6] - 2024-10-23

### Changed
- Updated cmaas_utils to version 2.0
- Updated cdr_schemas to version 4.9.0
- Mask and crop functions moved from this repo to cmaas_utils
- Updated mask and crop calls to use cmaas_utils version
- Changed internal segmentation mask format to support storing confidence and provenance.
- Updated segmentation step to support new layout format
- Updated generate_geometry step to new layout format
- Updated save_output step to support new segmentation mask format
- Updated validation step to support new segmentation mask format


## [0.4.5] - 2024-08-20

### Changed
- Update cmass_util to 0.1.14
- Changed ice-resin to icy_resin

## [0.4.4] - 2024-08-19

### Added
- Adding icy-resin model.
- ML based map unit detection model Yolo_Legends
- Added submodule containing support code for Yolo_Legends
- Added Yolo_Legends checkpoint file to docker container

### Changed
- Changed pipeline to use ML model for legend detection instead of heursitic method.
- Updated cmaas_utils to 0.1.12, need to update package to run.


## [0.4.3] - 2024-08-06

### Changed
- Changed json format of cdr data for amqp mode to support legend items
- Pinned numpy < 2
- Added `--checkpoint_dir` option to specify what directory to use for loading model checkpoints
- Changed the checkpoint directories structure and update paths in model interface files accordingly


## [0.4.2] - 2024-06-13

### Changed
- updated README.md to accommodate new users and developers
- When saving cdr_json it now uses `<name>-<model>:<version>` with just `-` and lower case.
- pinned matplotlib version to 3.8.4
- updated validation submodule head
- Changed filename sanitization to only allow alphanumeric characters and ' ', '-', '_' in names
- Added polygon legend mask to legend extraction step
- Added support for validating drab_volcano results
- Updated drab_volcano alias list
- Updated validation submodule head
- Updated validation pipeline step
- Fixed bug with duplicate legends in maps causing 0 validation scores for some legend items
- Added graph of validation scores for full dataset
- When saving cdr_json system_name is not changed, version has model appended

## [0.4.1] - 2024-05-09

### Changed
- When saving cdr_json, make `cog_id` default to string "-1"
- add model to cdr_system (for example uiuc_golden_muscat) when saving to cdr
- docker image will have the CDR_SYSTEM_VERSION set automatically based on tag/branch/pr

### Fixed
- Point features had a leading zero, e.g. [0.0, x, y]. Striped out the zero to make [x,y]
- saving raster for point features

## [0.4.0] - 2024-05-02

### Added
- segmentation data from CDR
- cog_id to cdr_json output
- `--cdr_system`  and `--cdr_system_version` args to set system and system_version in the cdr output respectively
- `--output_types` argument to pipeline, accepts cdr_json, geopackage, raster_masks. Default is cdr_json, geopackage

### Changed
- Refactor for multiprocessing workers
- Multi GPU support
- Bug fixes discovered during soak testing

## [0.3.0] - 2024-04-20

### Added
- Outputs are now saved using the CDR schema as well as tiff images
- New model (drab_volcano), this is not finished yet
- GitHub Action to build docker images

### Changed
- Git Submodules now point to either GitHub or public download locations

## [0.2.0] - 2024-04-14

RabbitMQ processing, this assumes a folder where data is downloaded externally when a new
map is available. See [uiuc-cdr](https://github.com/DARPA-CRITICALMAAS/uiuc-cdr).

### Added
- Can now use RabbitMQ queue to processes messages from CDR.

## [0.1.0] - 2024-04-10

This is the initial release of the code base. This will run on a folder with data and write
the results to an outputs folder
