# Changelog
All notable changes to this project will be documented in this file.

## [0.2.0] - 2020-05-13
### Changed
- Library structure changed to the tasks-oriented design

## [0.1.0] - 2020-05-11
### Added
- Training with meta text ([CTRL](https://arxiv.org/pdf/1909.05858.pdf)).

### Changed
- Global library structure refactored for further expansion.
- Data format has changed. Now no need for intermediate data arrays preparation.
Training is performed on raw text files. 
- Now data is processed during the training with multiprocessing workers.
- Embeddings now resized with mean embeddings vector (not the random one).

## [0.0.2]
### Added
- Added telegram client for text generator service.
- Added unlikelihood candidates loss.