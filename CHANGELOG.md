# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added
- Implement one-pass variance computation in golden_muscat inference code

## [0.4.2] - 2024-05-10

### Changed
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
