# EfficientDet for Edge-Aware Semantic Segmentation

Assignment for Object Detection during VR course at MIMUW in 2022, graded for `10 / 10`\
Beware as repo is ~400MB large, as it consists of two fully trained models\
Author: Marcin Mazur

## Code Structure

- `models` - best model checkpoints.
- `plots` - plots generated during trainings.
- `report` - report with techinques, summary of my work (`mm418420.pdf`).
- `src` source files.
  - `data` - module for data processing.
  - `utils` - module for useful help functions.
- `efficient_det.ipynb` - notebook with logs from best run.

## Performance

Tests should be fast ~1min, and can be run with `pytest` command.\
Full training (`10` epochs) takes ~3min on 4GB GPU.

## Environment

Project was developed on:

- python 3.9.7
- pytorch 1.11.0

## Task description

In this task - one should implement a part of the one of the state-of-the-art NN architecture for dense predictions - [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf) - called the BiFPN feature combinator. Then one should use the features from the FPN as an input to two heads: one for an person segmentation (single class) and other for an edge detection. This network should be trained on the PennFundan dataset with traditional augmentations.
