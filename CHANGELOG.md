# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- (Prefixed)ClusterHelper is only used for truly multi-source datasets now (atm it's only MovieGraphBenchmark's multi setting)

### Fixed

- Added left/right triples to MovieGraphBenchmark for consistency in binary case
- Addressed property/class links problem in OAEI

## [0.3.0] - 2024-03-25

### Added

- MED-BBK dataset
- Dataset statistics
- Support for multi-source datasets
- Multi-source case for MovieGraphBenchmark

### Changed

- entity links are now handled via eche's (Prefixed)ClusterHelper
- Very large and very small datasets only allow dask/pandas backend respectively

### Fixed

- dask/pandas backend typing

## [0.2.1] - 2023-08-09

### Fixed

- Fix oaei property/class links problem

## [0.2.0] - 2023-08-30

### Added

- Added id mapped dataset class
- Added canonical name property
- Added dask support
- Added dataset names
- Added caching functionality

### Fixed

- Fixed backend handling of folds

## [0.1.1] - 2023-03-27

### Fixed

- Loosen python dependency

[0.1.1]: https://github.com/dobraczka/sylloge/releases/tag/v0.1.1
[0.2.0]: https://github.com/dobraczka/sylloge/releases/tag/v0.2.0
[0.2.1]: https://github.com/dobraczka/sylloge/releases/tag/v0.2.1
