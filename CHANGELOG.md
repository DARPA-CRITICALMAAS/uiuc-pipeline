# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.2.0] - 2024-04-14

RabbitMQ processing, this assumes a folder where data is downloaded externally when a new
map is available. See [uiuc-cdr](https://github.com/DARPA-CRITICALMAAS/uiuc-cdr).

### Added
- Can now use RabbitMQ queue to processes messages from CDR.

## [0.1.0] - 2024-04-10

This is the initial release of the code base. This will run on a folder with data and write
the results to an outputs folder
